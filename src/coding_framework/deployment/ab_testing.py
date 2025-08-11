"""
A/B Testing framework for gradual rollout of VERL-trained models.
Provides controlled testing capabilities with traffic splitting and performance comparison.
"""

import asyncio
import time
import json
import random
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import structlog
import numpy as np
from pathlib import Path


class TestStatus(Enum):
    """Status of an A/B test."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class TestVariant(Enum):
    """Variants in an A/B test."""
    CONTROL = "control"  # Existing model
    TREATMENT = "treatment"  # New model
    

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    
    # Test identification
    test_id: str
    test_name: str
    description: str = ""
    
    # Models
    control_model_path: str = ""
    treatment_model_path: str = ""
    
    # Traffic configuration
    initial_traffic_split: float = 0.1  # Start with 10% to treatment
    max_traffic_split: float = 1.0  # Maximum traffic to treatment
    traffic_increment: float = 0.1  # Increment per successful interval
    
    # Duration settings
    min_test_duration_hours: float = 24.0
    max_test_duration_hours: float = 168.0  # 1 week
    evaluation_interval_minutes: float = 30.0
    
    # Success criteria
    success_metric: str = "accuracy"
    min_improvement: float = 0.01  # 1% improvement required
    confidence_level: float = 0.95
    min_samples_per_variant: int = 1000
    
    # Safety thresholds
    max_error_rate: float = 0.05
    max_latency_ms: float = 100.0
    rollback_on_degradation: bool = True
    
    # Monitoring
    enable_detailed_logging: bool = True
    metrics_collection_enabled: bool = True
    
    # Auto-scaling
    auto_scale_traffic: bool = True
    scale_based_on_performance: bool = True


@dataclass
class VariantMetrics:
    """Metrics collected for a test variant."""
    
    variant: TestVariant
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Model metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Business metrics
    conversion_rate: float = 0.0
    user_satisfaction: float = 0.0
    
    # Resource utilization
    avg_cpu_usage: float = 0.0
    avg_memory_usage_mb: float = 0.0
    avg_gpu_usage: float = 0.0
    
    # Time series data
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_latency_stats(self, latencies: List[float]):
        """Update latency statistics."""
        if latencies:
            self.avg_latency_ms = np.mean(latencies)
            self.p50_latency_ms = np.percentile(latencies, 50)
            self.p95_latency_ms = np.percentile(latencies, 95)
            self.p99_latency_ms = np.percentile(latencies, 99)


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    
    test_id: str
    test_status: TestStatus
    start_time: float
    end_time: Optional[float] = None
    
    # Metrics for each variant
    control_metrics: VariantMetrics = field(default_factory=lambda: VariantMetrics(TestVariant.CONTROL))
    treatment_metrics: VariantMetrics = field(default_factory=lambda: VariantMetrics(TestVariant.TREATMENT))
    
    # Statistical analysis
    statistical_significance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    effect_size: float = 0.0
    power_analysis: float = 0.0
    
    # Decision
    winner: Optional[TestVariant] = None
    recommendation: str = ""
    rollout_percentage: float = 0.0
    
    # Test progression
    traffic_split_history: List[Dict[str, Any]] = field(default_factory=list)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)


class ABTestManager:
    """
    Manages A/B testing for model deployments with gradual rollout capabilities.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.active_tests: Dict[str, ABTestResult] = {}
        self.completed_tests: List[ABTestResult] = []
        self.traffic_router = TrafficRouter()
        
    async def start_test(self, config: ABTestConfig) -> ABTestResult:
        """
        Start a new A/B test with the given configuration.
        
        Args:
            config: A/B test configuration
            
        Returns:
            ABTestResult tracking the test progress
        """
        self.logger.info(
            "Starting A/B test",
            test_id=config.test_id,
            initial_split=config.initial_traffic_split
        )
        
        # Create test result
        result = ABTestResult(
            test_id=config.test_id,
            test_status=TestStatus.RUNNING,
            start_time=time.time()
        )
        
        # Initialize traffic routing
        await self.traffic_router.configure(
            config.test_id,
            config.initial_traffic_split
        )
        
        # Store active test
        self.active_tests[config.test_id] = result
        
        # Start monitoring loop
        asyncio.create_task(self._monitor_test(config, result))
        
        return result
    
    async def _monitor_test(self, config: ABTestConfig, result: ABTestResult):
        """
        Monitor an active A/B test and make traffic decisions.
        
        Args:
            config: Test configuration
            result: Test result object to update
        """
        evaluation_interval = config.evaluation_interval_minutes * 60
        
        while result.test_status == TestStatus.RUNNING:
            try:
                await asyncio.sleep(evaluation_interval)
                
                # Collect metrics
                await self._collect_metrics(config, result)
                
                # Analyze performance
                analysis = self._analyze_performance(config, result)
                
                # Make traffic decision
                decision = await self._make_traffic_decision(config, result, analysis)
                
                # Log decision
                result.decision_log.append({
                    "timestamp": time.time(),
                    "analysis": analysis,
                    "decision": decision
                })
                
                # Check completion criteria
                if await self._should_complete_test(config, result):
                    await self._complete_test(config, result)
                    
            except Exception as e:
                self.logger.error(
                    "Error monitoring A/B test",
                    test_id=config.test_id,
                    error=str(e)
                )
                result.test_status = TestStatus.FAILED
    
    async def _collect_metrics(self, config: ABTestConfig, result: ABTestResult):
        """Collect metrics for both variants."""
        # This would integrate with your monitoring system
        # For now, using placeholder implementation
        pass
    
    def _analyze_performance(
        self,
        config: ABTestConfig, 
        result: ABTestResult
    ) -> Dict[str, Any]:
        """
        Analyze performance difference between variants.
        
        Returns:
            Analysis results including statistical significance
        """
        control = result.control_metrics
        treatment = result.treatment_metrics
        
        # Calculate performance difference
        metric_value_control = getattr(control, config.success_metric, 0)
        metric_value_treatment = getattr(treatment, config.success_metric, 0)
        
        if metric_value_control > 0:
            improvement = (metric_value_treatment - metric_value_control) / metric_value_control
        else:
            improvement = 0
        
        # Simple statistical significance calculation
        # In production, use proper statistical tests
        sample_size = min(control.request_count, treatment.request_count)
        if sample_size >= config.min_samples_per_variant:
            # Simplified significance calculation
            significance = min(0.99, sample_size / (config.min_samples_per_variant * 2))
        else:
            significance = 0
        
        return {
            "improvement": improvement,
            "significance": significance,
            "control_metric": metric_value_control,
            "treatment_metric": metric_value_treatment,
            "sample_size": sample_size
        }
    
    async def _make_traffic_decision(
        self,
        config: ABTestConfig,
        result: ABTestResult,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make traffic routing decision based on analysis.
        
        Returns:
            Decision including new traffic split
        """
        current_split = self.traffic_router.get_current_split(config.test_id)
        new_split = current_split
        action = "maintain"
        
        # Check for degradation
        if analysis["improvement"] < -0.05:  # 5% degradation
            if config.rollback_on_degradation:
                new_split = 0
                action = "rollback"
                result.test_status = TestStatus.ROLLED_BACK
        
        # Check for improvement with significance
        elif (analysis["improvement"] >= config.min_improvement and 
              analysis["significance"] >= config.confidence_level):
            
            if config.auto_scale_traffic:
                # Gradually increase traffic
                new_split = min(
                    config.max_traffic_split,
                    current_split + config.traffic_increment
                )
                action = "increase"
        
        # Apply new split if changed
        if new_split != current_split:
            await self.traffic_router.configure(config.test_id, new_split)
            result.rollout_percentage = new_split
        
        return {
            "action": action,
            "previous_split": current_split,
            "new_split": new_split,
            "reason": f"Improvement: {analysis['improvement']:.2%}, "
                     f"Significance: {analysis['significance']:.2f}"
        }
    
    async def _should_complete_test(
        self,
        config: ABTestConfig,
        result: ABTestResult
    ) -> bool:
        """Check if test should be completed."""
        
        # Check minimum duration
        duration_hours = (time.time() - result.start_time) / 3600
        if duration_hours < config.min_test_duration_hours:
            return False
        
        # Check maximum duration
        if duration_hours >= config.max_test_duration_hours:
            return True
        
        # Check if we have enough data
        control = result.control_metrics
        treatment = result.treatment_metrics
        
        if (control.request_count >= config.min_samples_per_variant and
            treatment.request_count >= config.min_samples_per_variant):
            
            # Check if we have clear winner
            if result.statistical_significance >= config.confidence_level:
                return True
        
        # Check if test was rolled back
        if result.test_status == TestStatus.ROLLED_BACK:
            return True
        
        return False
    
    async def _complete_test(self, config: ABTestConfig, result: ABTestResult):
        """Complete an A/B test and determine winner."""
        
        result.end_time = time.time()
        result.test_status = TestStatus.COMPLETED
        
        # Determine winner
        analysis = self._analyze_performance(config, result)
        
        if (analysis["improvement"] >= config.min_improvement and
            analysis["significance"] >= config.confidence_level):
            result.winner = TestVariant.TREATMENT
            result.recommendation = "Deploy treatment model"
            result.rollout_percentage = 1.0
        else:
            result.winner = TestVariant.CONTROL
            result.recommendation = "Keep control model"
            result.rollout_percentage = 0.0
        
        # Move to completed tests
        self.completed_tests.append(result)
        del self.active_tests[config.test_id]
        
        self.logger.info(
            "A/B test completed",
            test_id=config.test_id,
            winner=result.winner.value,
            recommendation=result.recommendation
        )
    
    async def stop_test(self, test_id: str) -> ABTestResult:
        """Manually stop an active test."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        result = self.active_tests[test_id]
        result.test_status = TestStatus.PAUSED
        result.end_time = time.time()
        
        # Reset traffic to control
        await self.traffic_router.configure(test_id, 0.0)
        
        return result
    
    def get_test_status(self, test_id: str) -> Optional[ABTestResult]:
        """Get current status of a test."""
        
        if test_id in self.active_tests:
            return self.active_tests[test_id]
        
        for test in self.completed_tests:
            if test.test_id == test_id:
                return test
        
        return None
    
    def get_active_tests(self) -> List[ABTestResult]:
        """Get all active tests."""
        return list(self.active_tests.values())


class TrafficRouter:
    """
    Routes traffic between control and treatment variants.
    """
    
    def __init__(self):
        self.traffic_splits: Dict[str, float] = {}
        self.logger = structlog.get_logger()
    
    async def configure(self, test_id: str, treatment_percentage: float):
        """
        Configure traffic split for a test.
        
        Args:
            test_id: Test identifier
            treatment_percentage: Percentage of traffic to route to treatment (0-1)
        """
        self.traffic_splits[test_id] = treatment_percentage
        
        self.logger.info(
            "Traffic split configured",
            test_id=test_id,
            treatment_percentage=treatment_percentage
        )
    
    def get_current_split(self, test_id: str) -> float:
        """Get current traffic split for a test."""
        return self.traffic_splits.get(test_id, 0.0)
    
    def route_request(self, test_id: str) -> TestVariant:
        """
        Determine which variant should handle a request.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Variant to route the request to
        """
        treatment_percentage = self.traffic_splits.get(test_id, 0.0)
        
        if random.random() < treatment_percentage:
            return TestVariant.TREATMENT
        else:
            return TestVariant.CONTROL
    
    def get_all_splits(self) -> Dict[str, float]:
        """Get all active traffic splits."""
        return self.traffic_splits.copy()