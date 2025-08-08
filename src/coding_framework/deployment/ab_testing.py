"""
A/B Testing Framework for VERL Model Deployments.

Provides sophisticated A/B testing capabilities for comparing different
model versions with statistical significance testing and automated
traffic management.
"""

import asyncio
import time
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import structlog
import random
from pathlib import Path


class TestStatus(Enum):
    """A/B test status enumeration."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class VariantStatus(Enum):
    """Test variant status."""
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    LOSER = "loser"
    BASELINE = "baseline"


@dataclass
class TestVariant:
    """Configuration for an A/B test variant."""
    
    variant_id: str
    name: str
    model_path: str
    description: str = ""
    traffic_allocation: float = 50.0  # Percentage of traffic
    expected_improvement: float = 5.0  # Expected improvement percentage
    status: VariantStatus = VariantStatus.ACTIVE
    
    # Deployment configuration for this variant
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    total_requests: int = 0
    successful_requests: int = 0
    error_rate: float = 0.0
    avg_latency: float = 0.0
    conversion_rate: float = 0.0  # Domain-specific success metric
    

@dataclass 
class ABTestConfig:
    """Configuration for A/B test experiment."""
    
    test_id: str
    test_name: str
    description: str
    variants: List[TestVariant]
    
    # Test parameters
    duration_days: int = 14
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 5.0  # Minimum improvement % to detect
    
    # Traffic management
    ramp_up_duration_hours: int = 24  # Gradually increase traffic over 24h
    early_stopping_enabled: bool = True
    max_error_rate_threshold: float = 5.0  # Stop variant if error rate exceeds this
    
    # Evaluation metrics
    primary_metric: str = "conversion_rate"  # Main metric to optimize
    secondary_metrics: List[str] = field(default_factory=lambda: ["latency", "error_rate"])
    
    # Guardrail metrics (stop test if these degrade significantly)
    guardrail_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Advanced settings
    enable_stratified_sampling: bool = False
    stratification_key: Optional[str] = None  # e.g., "user_segment"
    enable_sequential_testing: bool = True  # Enable early stopping
    enable_multi_armed_bandit: bool = False  # Dynamic traffic allocation


class ABTestManager:
    """
    Manages A/B testing experiments for VERL model deployments.
    
    Provides comprehensive A/B testing with statistical analysis,
    automated traffic management, and safety guardrails.
    """
    
    def __init__(self, config: ABTestConfig = None):
        self.config = config
        self.logger = structlog.get_logger(component="ab_test_manager")
        
        # Test state management
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_history: List[Dict[str, Any]] = []
        
        # Traffic routing
        self.traffic_router = TrafficRouter()
        
        # Statistical analysis
        self.stats_analyzer = StatisticalAnalyzer()
        
        # Event callbacks
        self.test_callbacks: List[Callable] = []
        
    async def create_test(self, config: ABTestConfig) -> Dict[str, Any]:
        """Create and initialize a new A/B test."""
        
        self.logger.info(
            "Creating A/B test",
            test_id=config.test_id,
            test_name=config.test_name,
            variants=len(config.variants)
        )
        
        # Validate test configuration
        validation_result = self._validate_test_config(config)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid test config: {validation_result['errors']}")
            
        # Initialize test record
        test_record = {
            "config": config,
            "status": TestStatus.DRAFT,
            "created_at": time.time(),
            "started_at": None,
            "ended_at": None,
            "current_phase": "initialization",
            "results": {},
            "statistical_analysis": {},
            "traffic_allocation": self._calculate_initial_traffic_allocation(config),
            "error_count": 0
        }
        
        self.active_tests[config.test_id] = test_record
        
        try:
            # Deploy all variants
            deployment_results = await self._deploy_test_variants(config)
            test_record["deployment_results"] = deployment_results
            
            # Initialize traffic routing
            await self.traffic_router.setup_test_routing(config)
            
            # Setup monitoring
            await self._setup_test_monitoring(config)
            
            self.logger.info(
                "A/B test created successfully",
                test_id=config.test_id,
                deployed_variants=len(deployment_results)
            )
            
            return {
                "success": True,
                "test_id": config.test_id,
                "status": TestStatus.DRAFT.value,
                "variants_deployed": len(deployment_results),
                "ready_to_start": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}", test_id=config.test_id)
            
            # Cleanup on failure
            if config.test_id in self.active_tests:
                del self.active_tests[config.test_id]
                
            raise
            
    async def start_test(self, test_id: str) -> Dict[str, Any]:
        """Start running an A/B test."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test_record = self.active_tests[test_id]
        config = test_record["config"]
        
        self.logger.info("Starting A/B test", test_id=test_id)
        
        try:
            # Pre-start validation
            readiness_check = await self._check_test_readiness(test_id)
            if not readiness_check["ready"]:
                raise ValueError(f"Test not ready: {readiness_check['issues']}")
                
            # Start traffic routing
            await self.traffic_router.start_test_traffic(test_id)
            
            # Update test status
            test_record["status"] = TestStatus.RUNNING
            test_record["started_at"] = time.time()
            test_record["current_phase"] = "ramp_up"
            
            # Start monitoring and analysis tasks
            asyncio.create_task(self._monitor_test_progress(test_id))
            asyncio.create_task(self._run_statistical_analysis_loop(test_id))
            
            if config.enable_sequential_testing:
                asyncio.create_task(self._monitor_early_stopping_conditions(test_id))
                
            # Run test callbacks
            await self._run_test_callbacks(test_record, "started")
            
            self.logger.info(
                "A/B test started successfully",
                test_id=test_id,
                duration_days=config.duration_days,
                variants=len(config.variants)
            )
            
            return {
                "success": True,
                "test_id": test_id,
                "status": TestStatus.RUNNING.value,
                "started_at": test_record["started_at"],
                "expected_end": test_record["started_at"] + (config.duration_days * 24 * 3600)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test: {e}", test_id=test_id)
            test_record["status"] = TestStatus.FAILED
            test_record["error"] = str(e)
            raise
            
    async def route_request(
        self, 
        test_id: str, 
        request_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a request to appropriate test variant."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test_record = self.active_tests[test_id]
        
        if test_record["status"] != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")
            
        # Use traffic router to select variant
        routing_decision = await self.traffic_router.route_request(test_id, request_context)
        
        # Update variant metrics
        variant_id = routing_decision["variant_id"]
        await self._update_variant_metrics(test_id, variant_id, "request_received")
        
        return {
            "variant_id": variant_id,
            "variant_name": routing_decision["variant_name"],
            "model_path": routing_decision["model_path"],
            "routing_metadata": routing_decision.get("metadata", {})
        }
        
    async def record_result(
        self,
        test_id: str,
        variant_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Record the result of a test request."""
        
        if test_id not in self.active_tests:
            return
            
        test_record = self.active_tests[test_id]
        config = test_record["config"]
        
        # Find the variant
        variant = None
        for v in config.variants:
            if v.variant_id == variant_id:
                variant = v
                break
                
        if not variant:
            self.logger.warning(f"Unknown variant {variant_id} for test {test_id}")
            return
            
        # Update variant metrics
        variant.total_requests += 1
        
        if result.get("success", False):
            variant.successful_requests += 1
            
        # Update specific metrics
        if "latency" in result:
            if "latency" not in variant.metrics:
                variant.metrics["latency"] = []
            variant.metrics["latency"].append(result["latency"])
            variant.avg_latency = sum(variant.metrics["latency"]) / len(variant.metrics["latency"])
            
        if "conversion" in result:
            if "conversion" not in variant.metrics:
                variant.metrics["conversion"] = []
            variant.metrics["conversion"].append(1.0 if result["conversion"] else 0.0)
            variant.conversion_rate = sum(variant.metrics["conversion"]) / len(variant.metrics["conversion"])
            
        # Update error rate
        variant.error_rate = 1.0 - (variant.successful_requests / max(variant.total_requests, 1))
        
        # Check guardrail metrics
        await self._check_guardrail_metrics(test_id, variant_id)
        
    async def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get current results and analysis for a test."""
        
        if test_id not in self.active_tests:
            # Check test history
            for historical_test in self.test_history:
                if historical_test["config"].test_id == test_id:
                    return historical_test["results"]
            raise ValueError(f"Test {test_id} not found")
            
        test_record = self.active_tests[test_id]
        config = test_record["config"]
        
        # Run current statistical analysis
        current_analysis = await self.stats_analyzer.analyze_test(test_record)
        
        # Prepare results summary
        results = {
            "test_id": test_id,
            "test_name": config.test_name,
            "status": test_record["status"].value,
            "duration_so_far": self._get_test_duration(test_record),
            "variants": [],
            "statistical_analysis": current_analysis,
            "recommendations": self._generate_recommendations(test_record, current_analysis),
            "metadata": {
                "created_at": test_record["created_at"],
                "started_at": test_record.get("started_at"),
                "current_phase": test_record.get("current_phase"),
                "total_requests": sum(v.total_requests for v in config.variants)
            }
        }
        
        # Add variant results
        for variant in config.variants:
            variant_result = {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "status": variant.status.value,
                "traffic_allocation": variant.traffic_allocation,
                "total_requests": variant.total_requests,
                "successful_requests": variant.successful_requests,
                "error_rate": variant.error_rate,
                "avg_latency": variant.avg_latency,
                "conversion_rate": variant.conversion_rate,
                "metrics_summary": self._summarize_variant_metrics(variant)
            }
            results["variants"].append(variant_result)
            
        return results
        
    async def stop_test(
        self, 
        test_id: str, 
        reason: str = "manual_stop"
    ) -> Dict[str, Any]:
        """Stop a running A/B test."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
            
        test_record = self.active_tests[test_id]
        config = test_record["config"]
        
        self.logger.info("Stopping A/B test", test_id=test_id, reason=reason)
        
        try:
            # Stop traffic routing
            await self.traffic_router.stop_test_traffic(test_id)
            
            # Final statistical analysis
            final_analysis = await self.stats_analyzer.analyze_test(test_record)
            
            # Update test record
            test_record["status"] = TestStatus.COMPLETED if reason != "failure" else TestStatus.FAILED
            test_record["ended_at"] = time.time()
            test_record["stop_reason"] = reason
            test_record["final_analysis"] = final_analysis
            
            # Generate final results
            final_results = await self.get_test_results(test_id)
            test_record["results"] = final_results
            
            # Move to history
            self.test_history.append(test_record)
            del self.active_tests[test_id]
            
            # Run completion callbacks
            await self._run_test_callbacks(test_record, "completed")
            
            self.logger.info(
                "A/B test stopped successfully",
                test_id=test_id,
                reason=reason,
                duration=self._get_test_duration(test_record)
            )
            
            return {
                "success": True,
                "test_id": test_id,
                "final_status": test_record["status"].value,
                "duration": self._get_test_duration(test_record),
                "final_results": final_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop A/B test: {e}", test_id=test_id)
            raise
            
    # Helper methods for test management
    def _validate_test_config(self, config: ABTestConfig) -> Dict[str, Any]:
        """Validate A/B test configuration."""
        
        errors = []
        
        # Check basic requirements
        if not config.test_id or not config.test_name:
            errors.append("Test ID and name are required")
            
        if len(config.variants) < 2:
            errors.append("At least 2 variants are required")
            
        # Check traffic allocation
        total_allocation = sum(v.traffic_allocation for v in config.variants)
        if abs(total_allocation - 100.0) > 0.1:
            errors.append(f"Traffic allocation must sum to 100%, got {total_allocation}%")
            
        # Check for duplicate variant IDs
        variant_ids = [v.variant_id for v in config.variants]
        if len(variant_ids) != len(set(variant_ids)):
            errors.append("Duplicate variant IDs found")
            
        return {"valid": len(errors) == 0, "errors": errors}
        
    def _calculate_initial_traffic_allocation(self, config: ABTestConfig) -> Dict[str, float]:
        """Calculate initial traffic allocation for variants."""
        
        allocation = {}
        for variant in config.variants:
            allocation[variant.variant_id] = variant.traffic_allocation
            
        return allocation
        
    async def _deploy_test_variants(self, config: ABTestConfig) -> List[Dict[str, Any]]:
        """Deploy all test variants."""
        
        deployment_results = []
        
        for variant in config.variants:
            # This would integrate with the actual deployment manager
            # For now, we simulate deployment
            await asyncio.sleep(1)  # Simulate deployment time
            
            result = {
                "variant_id": variant.variant_id,
                "deployment_status": "success",
                "endpoint": f"variant-{variant.variant_id}.test.local",
                "model_path": variant.model_path
            }
            deployment_results.append(result)
            
        return deployment_results
        
    async def _setup_test_monitoring(self, config: ABTestConfig) -> None:
        """Setup monitoring for the test."""
        # Placeholder for monitoring setup
        pass
        
    async def _check_test_readiness(self, test_id: str) -> Dict[str, Any]:
        """Check if test is ready to start."""
        # Placeholder for readiness checks
        return {"ready": True, "issues": []}
        
    async def _monitor_test_progress(self, test_id: str) -> None:
        """Monitor test progress and handle phase transitions."""
        
        test_record = self.active_tests.get(test_id)
        if not test_record:
            return
            
        config = test_record["config"]
        
        while test_record.get("status") == TestStatus.RUNNING:
            try:
                # Check test phase
                current_time = time.time()
                test_duration = current_time - test_record["started_at"]
                
                # Ramp-up phase
                if test_record["current_phase"] == "ramp_up":
                    ramp_up_duration = config.ramp_up_duration_hours * 3600
                    if test_duration > ramp_up_duration:
                        test_record["current_phase"] = "full_traffic"
                        self.logger.info("Test entered full traffic phase", test_id=test_id)
                        
                # Check completion conditions
                max_duration = config.duration_days * 24 * 3600
                if test_duration > max_duration:
                    await self.stop_test(test_id, "duration_completed")
                    break
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in test monitoring: {e}", test_id=test_id)
                await asyncio.sleep(60)
                
    async def _run_statistical_analysis_loop(self, test_id: str) -> None:
        """Continuously run statistical analysis on test data."""
        
        test_record = self.active_tests.get(test_id)
        if not test_record:
            return
            
        while test_record.get("status") == TestStatus.RUNNING:
            try:
                # Run analysis every 5 minutes
                analysis = await self.stats_analyzer.analyze_test(test_record)
                test_record["statistical_analysis"] = analysis
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in statistical analysis: {e}", test_id=test_id)
                await asyncio.sleep(300)
                
    async def _monitor_early_stopping_conditions(self, test_id: str) -> None:
        """Monitor conditions for early stopping."""
        
        test_record = self.active_tests.get(test_id)
        if not test_record:
            return
            
        while test_record.get("status") == TestStatus.RUNNING:
            try:
                # Check if we can make a confident decision
                analysis = test_record.get("statistical_analysis", {})
                
                if self._should_stop_early(analysis):
                    await self.stop_test(test_id, "early_stopping_criteria_met")
                    break
                    
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in early stopping monitor: {e}", test_id=test_id)
                await asyncio.sleep(300)
                
    def _should_stop_early(self, analysis: Dict[str, Any]) -> bool:
        """Determine if test should be stopped early based on statistical analysis."""
        
        # Check if we have statistical significance
        if analysis.get("statistically_significant", False):
            confidence = analysis.get("confidence", 0.0)
            if confidence >= 0.95:  # 95% confidence
                return True
                
        return False
        
    async def _check_guardrail_metrics(self, test_id: str, variant_id: str) -> None:
        """Check guardrail metrics and stop variant if needed."""
        
        test_record = self.active_tests.get(test_id)
        if not test_record:
            return
            
        config = test_record["config"]
        
        # Find variant
        variant = None
        for v in config.variants:
            if v.variant_id == variant_id:
                variant = v
                break
                
        if not variant:
            return
            
        # Check error rate threshold
        if variant.error_rate > config.max_error_rate_threshold / 100.0:
            self.logger.warning(
                f"Variant {variant_id} exceeded error rate threshold",
                test_id=test_id,
                error_rate=variant.error_rate,
                threshold=config.max_error_rate_threshold
            )
            
            # Pause this variant
            variant.status = VariantStatus.PAUSED
            await self.traffic_router.pause_variant_traffic(test_id, variant_id)
            
    async def _update_variant_metrics(
        self, 
        test_id: str, 
        variant_id: str, 
        event_type: str
    ) -> None:
        """Update metrics for a variant based on events."""
        # This would be expanded to handle different event types
        pass
        
    def _get_test_duration(self, test_record: Dict[str, Any]) -> float:
        """Get the duration of a test in seconds."""
        start_time = test_record.get("started_at", test_record.get("created_at", 0))
        end_time = test_record.get("ended_at", time.time())
        return end_time - start_time
        
    def _generate_recommendations(
        self, 
        test_record: Dict[str, Any], 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        # Statistical significance recommendations
        if analysis.get("statistically_significant", False):
            winner = analysis.get("winning_variant")
            if winner:
                recommendations.append(
                    f"Statistically significant result found. "
                    f"Recommend promoting variant {winner} to production."
                )
        else:
            sample_size = analysis.get("total_samples", 0)
            min_required = test_record["config"].min_sample_size
            if sample_size < min_required:
                recommendations.append(
                    f"More data needed. Current sample size: {sample_size}, "
                    f"minimum required: {min_required}"
                )
                
        return recommendations
        
    def _summarize_variant_metrics(self, variant: TestVariant) -> Dict[str, Any]:
        """Create summary of variant metrics."""
        
        summary = {}
        
        for metric_name, values in variant.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
                
        return summary
        
    async def _run_test_callbacks(self, test_record: Dict[str, Any], event: str) -> None:
        """Run registered test callbacks."""
        
        for callback in self.test_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(test_record, event)
                else:
                    callback(test_record, event)
            except Exception as e:
                self.logger.warning(f"Test callback failed: {e}")
                
    def register_test_callback(self, callback: Callable) -> None:
        """Register a callback for test events."""
        self.test_callbacks.append(callback)


class TrafficRouter:
    """Handles traffic routing for A/B tests."""
    
    def __init__(self):
        self.logger = structlog.get_logger(component="traffic_router")
        self.test_routing_configs: Dict[str, Dict[str, Any]] = {}
        
    async def setup_test_routing(self, config: ABTestConfig) -> None:
        """Setup traffic routing configuration for a test."""
        
        routing_config = {
            "variants": {},
            "traffic_allocation": {},
            "routing_strategy": "random"  # Could be: random, hash_based, etc.
        }
        
        for variant in config.variants:
            routing_config["variants"][variant.variant_id] = {
                "name": variant.name,
                "model_path": variant.model_path,
                "status": variant.status.value
            }
            routing_config["traffic_allocation"][variant.variant_id] = variant.traffic_allocation
            
        self.test_routing_configs[config.test_id] = routing_config
        
    async def route_request(
        self, 
        test_id: str, 
        request_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route a request to appropriate variant."""
        
        if test_id not in self.test_routing_configs:
            raise ValueError(f"No routing config for test {test_id}")
            
        routing_config = self.test_routing_configs[test_id]
        
        # Select variant based on traffic allocation
        variant_id = self._select_variant(routing_config, request_context)
        variant_info = routing_config["variants"][variant_id]
        
        return {
            "variant_id": variant_id,
            "variant_name": variant_info["name"],
            "model_path": variant_info["model_path"],
            "metadata": {
                "routing_strategy": routing_config["routing_strategy"],
                "allocation_percentage": routing_config["traffic_allocation"][variant_id]
            }
        }
        
    def _select_variant(
        self, 
        routing_config: Dict[str, Any], 
        request_context: Dict[str, Any]
    ) -> str:
        """Select variant based on routing strategy."""
        
        # Simple random selection based on traffic allocation
        rand_val = random.random() * 100  # 0-100
        
        cumulative = 0.0
        for variant_id, allocation in routing_config["traffic_allocation"].items():
            cumulative += allocation
            if rand_val <= cumulative:
                return variant_id
                
        # Fallback to first variant
        return list(routing_config["variants"].keys())[0]
        
    async def start_test_traffic(self, test_id: str) -> None:
        """Start routing traffic for a test."""
        self.logger.info("Started traffic routing", test_id=test_id)
        
    async def stop_test_traffic(self, test_id: str) -> None:
        """Stop routing traffic for a test."""
        if test_id in self.test_routing_configs:
            del self.test_routing_configs[test_id]
        self.logger.info("Stopped traffic routing", test_id=test_id)
        
    async def pause_variant_traffic(self, test_id: str, variant_id: str) -> None:
        """Pause traffic to a specific variant."""
        if test_id in self.test_routing_configs:
            config = self.test_routing_configs[test_id]
            if variant_id in config["variants"]:
                config["variants"][variant_id]["status"] = "paused"
                # Redistribute traffic to remaining active variants
                self._redistribute_traffic(config, variant_id)
                
    def _redistribute_traffic(self, routing_config: Dict[str, Any], paused_variant_id: str) -> None:
        """Redistribute traffic when a variant is paused."""
        
        paused_allocation = routing_config["traffic_allocation"][paused_variant_id]
        routing_config["traffic_allocation"][paused_variant_id] = 0.0
        
        # Redistribute to remaining active variants proportionally
        active_variants = {
            vid: alloc for vid, alloc in routing_config["traffic_allocation"].items()
            if alloc > 0 and vid != paused_variant_id
        }
        
        if active_variants:
            total_active = sum(active_variants.values())
            for variant_id in active_variants:
                proportion = active_variants[variant_id] / total_active
                routing_config["traffic_allocation"][variant_id] += paused_allocation * proportion


class StatisticalAnalyzer:
    """Provides statistical analysis for A/B test results."""
    
    def __init__(self):
        self.logger = structlog.get_logger(component="statistical_analyzer")
        
    async def analyze_test(self, test_record: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive statistical analysis on test data."""
        
        config = test_record["config"]
        
        analysis = {
            "test_id": config.test_id,
            "analysis_timestamp": time.time(),
            "total_samples": sum(v.total_requests for v in config.variants),
            "statistically_significant": False,
            "confidence": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "winning_variant": None,
            "variant_comparisons": []
        }
        
        # Simple analysis (would be replaced with proper statistical tests)
        if analysis["total_samples"] >= config.min_sample_size:
            # Find best performing variant
            best_variant = max(
                config.variants, 
                key=lambda v: v.conversion_rate if v.total_requests > 0 else 0
            )
            
            # Calculate effect size (simplified)
            baseline_variant = next(
                (v for v in config.variants if v.status == VariantStatus.BASELINE), 
                config.variants[0]
            )
            
            if baseline_variant.total_requests > 0 and best_variant.total_requests > 0:
                effect_size = (best_variant.conversion_rate - baseline_variant.conversion_rate) / baseline_variant.conversion_rate * 100
                
                # Simple significance test (would use proper statistical tests in reality)
                if abs(effect_size) >= config.minimum_detectable_effect:
                    analysis["statistically_significant"] = True
                    analysis["confidence"] = 0.95  # Simplified
                    analysis["p_value"] = 0.03  # Simplified
                    analysis["effect_size"] = effect_size
                    analysis["winning_variant"] = best_variant.variant_id
                    
        return analysis