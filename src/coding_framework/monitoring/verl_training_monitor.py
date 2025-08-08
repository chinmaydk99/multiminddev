"""
VERL Training Monitor - Comprehensive monitoring for VERL distributed training.

Provides real-time monitoring, metrics collection, and observability for
VERL-based RL training processes integrated with Ray clusters.
"""

import asyncio
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pathlib import Path
import psutil
import ray

from ..verl_integration.verl_config import VERLDistributedConfig


class TrainingPhase(Enum):
    """Training phases for monitoring."""
    INITIALIZATION = "initialization"
    POLICY_TRAINING = "policy_training"
    REWARD_MODELING = "reward_modeling"
    EVALUATION = "evaluation"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class MetricSeverity(Enum):
    """Severity levels for metrics and alerts."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    
    timestamp: float
    training_step: int
    phase: TrainingPhase
    
    # Core RL metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    episode_length_mean: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0
    
    # Performance metrics
    samples_per_second: float = 0.0
    steps_per_second: float = 0.0
    tokens_per_second: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    
    # Distributed training metrics
    cluster_nodes_active: int = 1
    gradient_sync_time: float = 0.0
    communication_overhead: float = 0.0
    
    # VERL-specific metrics
    verl_algorithm: str = "ppo"
    inference_time: float = 0.0
    model_loading_time: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    

@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    
    rule_id: str
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: float
    severity: MetricSeverity
    enabled: bool = True
    consecutive_violations: int = 3  # Alert after N consecutive violations
    cooldown_seconds: int = 300  # 5 minutes between same alerts


@dataclass
class VERLTrainingMonitorConfig:
    """Configuration for VERL training monitoring."""
    
    # Collection settings
    metrics_collection_interval: int = 30  # seconds
    detailed_logging_interval: int = 300   # 5 minutes
    checkpoint_monitoring: bool = True
    
    # Storage settings
    metrics_retention_hours: int = 168  # 1 week
    logs_retention_hours: int = 72      # 3 days
    export_metrics_to_file: bool = True
    metrics_export_path: str = "./training_metrics"
    
    # Ray cluster monitoring
    monitor_ray_cluster: bool = True
    ray_dashboard_integration: bool = True
    
    # Alert settings
    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["console", "file"])
    
    # Performance monitoring
    profile_memory_usage: bool = True
    profile_gpu_usage: bool = True
    monitor_distributed_metrics: bool = True
    
    # Integration settings
    tensorboard_integration: bool = False
    wandb_integration: bool = False
    mlflow_integration: bool = False


class VERLTrainingMonitor:
    """
    Comprehensive monitoring system for VERL distributed training.
    
    Provides real-time monitoring, alerting, and observability for
    VERL-based RL training with Ray cluster integration.
    """
    
    def __init__(self, config: VERLTrainingMonitorConfig = None):
        self.config = config or VERLTrainingMonitorConfig()
        self.logger = structlog.get_logger(component="verl_training_monitor")
        
        # Monitoring state
        self.monitoring_active = False
        self.training_start_time: Optional[float] = None
        self.current_phase = TrainingPhase.INITIALIZATION
        
        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.metrics_lock = threading.Lock()
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_violations: Dict[str, int] = {}  # Track consecutive violations
        self.last_alert_times: Dict[str, float] = {}
        
        # Performance tracking
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detection_enabled = True
        
        # Integration clients
        self.ray_client: Optional[ray.ClientBuilder] = None
        self.tensorboard_writer = None
        self.wandb_run = None
        
        # Callbacks
        self.metric_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
    async def start_monitoring(
        self, 
        training_config: VERLDistributedConfig = None
    ) -> Dict[str, Any]:
        """Start comprehensive training monitoring."""
        
        self.logger.info("Starting VERL training monitoring")
        
        self.monitoring_active = True
        self.training_start_time = time.time()
        
        # Setup Ray cluster monitoring if enabled
        if self.config.monitor_ray_cluster:
            await self._setup_ray_monitoring()
            
        # Setup external integrations
        await self._setup_external_integrations()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_monitoring_loop())
        asyncio.create_task(self._performance_analysis_loop())
        
        if self.config.checkpoint_monitoring:
            asyncio.create_task(self._checkpoint_monitoring_loop())
            
        self.logger.info(
            "VERL training monitoring started",
            metrics_interval=self.config.metrics_collection_interval,
            ray_monitoring=self.config.monitor_ray_cluster,
            alerts_enabled=self.config.enable_alerts
        )
        
        return {
            "success": True,
            "monitoring_started_at": self.training_start_time,
            "config": self.config.__dict__,
            "integrations": self._get_active_integrations()
        }
        
    async def record_training_metrics(
        self, 
        metrics: Dict[str, Any],
        training_step: int,
        phase: TrainingPhase = None
    ) -> None:
        """Record training metrics."""
        
        if not self.monitoring_active:
            return
            
        # Update current phase if provided
        if phase:
            self.current_phase = phase
            
        # Create structured metrics object
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            training_step=training_step,
            phase=self.current_phase,
            
            # Extract core RL metrics
            policy_loss=metrics.get("policy_loss", 0.0),
            value_loss=metrics.get("value_loss", 0.0),
            reward_mean=metrics.get("reward_mean", 0.0),
            reward_std=metrics.get("reward_std", 0.0),
            episode_length_mean=metrics.get("episode_length_mean", 0.0),
            kl_divergence=metrics.get("kl_divergence", 0.0),
            entropy=metrics.get("entropy", 0.0),
            
            # Extract performance metrics
            samples_per_second=metrics.get("samples_per_second", 0.0),
            steps_per_second=metrics.get("steps_per_second", 0.0),
            tokens_per_second=metrics.get("tokens_per_second", 0.0),
            gpu_utilization=metrics.get("gpu_utilization", 0.0),
            memory_usage_gb=metrics.get("memory_usage_gb", 0.0),
            
            # Extract distributed metrics
            cluster_nodes_active=metrics.get("cluster_nodes_active", 1),
            gradient_sync_time=metrics.get("gradient_sync_time", 0.0),
            communication_overhead=metrics.get("communication_overhead", 0.0),
            
            # VERL-specific metrics
            verl_algorithm=metrics.get("verl_algorithm", "ppo"),
            inference_time=metrics.get("inference_time", 0.0),
            model_loading_time=metrics.get("model_loading_time", 0.0),
            
            # Custom metrics
            custom_metrics={k: v for k, v in metrics.items() 
                          if k not in self._get_standard_metric_names()}
        )
        
        # Store metrics
        with self.metrics_lock:
            self.metrics_history.append(training_metrics)
            
            # Maintain retention policy
            if self.config.metrics_retention_hours > 0:
                cutoff_time = time.time() - (self.config.metrics_retention_hours * 3600)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
        # Run metric callbacks
        await self._run_metric_callbacks(training_metrics)
        
        # Export to external systems
        await self._export_metrics(training_metrics)
        
        self.logger.debug(
            "Training metrics recorded",
            step=training_step,
            phase=phase.value if phase else self.current_phase.value,
            reward_mean=training_metrics.reward_mean,
            policy_loss=training_metrics.policy_loss
        )
        
    async def record_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        training_step: int
    ) -> None:
        """Record evaluation results from benchmark evaluations."""
        
        if not self.monitoring_active:
            return
            
        self.logger.info(
            "Recording evaluation results",
            step=training_step,
            benchmarks=list(evaluation_results.get("benchmarks", {}).keys())
        )
        
        # Extract evaluation metrics
        eval_metrics = {
            "evaluation_step": training_step,
            "evaluation_timestamp": time.time()
        }
        
        # Process benchmark results
        benchmarks = evaluation_results.get("benchmarks", {})
        for benchmark_name, results in benchmarks.items():
            if isinstance(results, dict):
                eval_metrics[f"{benchmark_name}_pass_at_1"] = results.get("pass_at_1", 0.0)
                eval_metrics[f"{benchmark_name}_pass_at_10"] = results.get("pass_at_10", 0.0)
                eval_metrics[f"{benchmark_name}_solved"] = results.get("solved_problems", 0)
                eval_metrics[f"{benchmark_name}_total"] = results.get("total_problems", 0)
                
        # Process aggregate results
        aggregate = evaluation_results.get("aggregate", {})
        if aggregate:
            eval_metrics["weighted_pass_at_1"] = aggregate.get("weighted_pass_at_1", 0.0)
            eval_metrics["overall_pass_rate"] = aggregate.get("overall_pass_rate", 0.0)
            eval_metrics["total_problems_evaluated"] = aggregate.get("total_problems", 0)
            
        # Record as training metrics with evaluation phase
        await self.record_training_metrics(
            eval_metrics,
            training_step,
            TrainingPhase.EVALUATION
        )
        
    async def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary and statistics."""
        
        if not self.metrics_history:
            return {"error": "No metrics recorded yet"}
            
        with self.metrics_lock:
            latest_metrics = self.metrics_history[-1]
            
            # Calculate training duration
            training_duration = (time.time() - self.training_start_time) if self.training_start_time else 0
            
            # Calculate performance statistics
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
            
            avg_reward = sum(m.reward_mean for m in recent_metrics) / len(recent_metrics)
            avg_policy_loss = sum(m.policy_loss for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.samples_per_second for m in recent_metrics) / len(recent_metrics)
            avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            # Phase distribution
            phase_counts = {}
            for metric in self.metrics_history:
                phase = metric.phase.value
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
            # Alert statistics
            alert_stats = {
                "total_alerts": len(self.alert_violations),
                "active_violations": sum(1 for v in self.alert_violations.values() if v > 0),
                "alert_rules_count": len(self.alert_rules)
            }
            
            summary = {
                "training_status": {
                    "monitoring_active": self.monitoring_active,
                    "current_phase": self.current_phase.value,
                    "latest_training_step": latest_metrics.training_step,
                    "training_duration_hours": training_duration / 3600,
                    "total_metrics_recorded": len(self.metrics_history)
                },
                "performance_summary": {
                    "avg_reward_recent": avg_reward,
                    "avg_policy_loss_recent": avg_policy_loss,
                    "avg_throughput_recent": avg_throughput,
                    "avg_gpu_utilization_recent": avg_gpu_util,
                    "latest_kl_divergence": latest_metrics.kl_divergence,
                    "latest_entropy": latest_metrics.entropy
                },
                "distributed_training": {
                    "cluster_nodes_active": latest_metrics.cluster_nodes_active,
                    "gradient_sync_time": latest_metrics.gradient_sync_time,
                    "communication_overhead": latest_metrics.communication_overhead
                },
                "phase_distribution": phase_counts,
                "alert_statistics": alert_stats,
                "integrations": self._get_active_integrations(),
                "last_updated": latest_metrics.timestamp
            }
            
            return summary
            
    async def get_performance_analysis(self) -> Dict[str, Any]:
        """Get detailed performance analysis and recommendations."""
        
        if not self.metrics_history:
            return {"error": "Insufficient data for analysis"}
            
        analysis = {
            "training_efficiency": self._analyze_training_efficiency(),
            "resource_utilization": self._analyze_resource_utilization(), 
            "distributed_performance": self._analyze_distributed_performance(),
            "convergence_analysis": self._analyze_convergence(),
            "anomalies_detected": self._detect_anomalies(),
            "recommendations": self._generate_performance_recommendations()
        }
        
        return analysis
        
    def add_alert_rule(self, alert_rule: AlertRule) -> None:
        """Add a custom alert rule."""
        
        self.alert_rules[alert_rule.rule_id] = alert_rule
        self.alert_violations[alert_rule.rule_id] = 0
        
        self.logger.info(
            "Alert rule added",
            rule_id=alert_rule.rule_id,
            metric=alert_rule.metric_name,
            threshold=alert_rule.threshold,
            severity=alert_rule.severity.value
        )
        
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            if rule_id in self.alert_violations:
                del self.alert_violations[rule_id]
            if rule_id in self.last_alert_times:
                del self.last_alert_times[rule_id]
            return True
        return False
        
    # Internal monitoring methods
    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                
                # Collect Ray cluster metrics if enabled
                if self.config.monitor_ray_cluster:
                    ray_metrics = await self._collect_ray_cluster_metrics()
                    system_metrics.update(ray_metrics)
                    
                # Record system metrics
                if system_metrics:
                    await self.record_training_metrics(
                        system_metrics,
                        training_step=0,  # System metrics don't have training steps
                        phase=self.current_phase
                    )
                    
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)
                
    async def _alert_monitoring_loop(self) -> None:
        """Alert monitoring and evaluation loop."""
        
        while self.monitoring_active:
            try:
                if self.config.enable_alerts and self.metrics_history:
                    await self._evaluate_alert_rules()
                    
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(30)
                
    async def _performance_analysis_loop(self) -> None:
        """Performance analysis loop."""
        
        while self.monitoring_active:
            try:
                # Run performance analysis every 5 minutes
                if len(self.metrics_history) >= 10:  # Need minimum data
                    analysis = await self.get_performance_analysis()
                    
                    # Log key insights
                    if "recommendations" in analysis:
                        for rec in analysis["recommendations"][:3]:  # Top 3 recommendations
                            self.logger.info(f"Performance recommendation: {rec}")
                            
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance analysis loop: {e}")
                await asyncio.sleep(300)
                
    async def _checkpoint_monitoring_loop(self) -> None:
        """Monitor checkpoint creation and validation."""
        
        while self.monitoring_active:
            try:
                # Monitor checkpoint directory and log checkpoint events
                # This would integrate with VERL's checkpoint system
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in checkpoint monitoring loop: {e}")
                await asyncio.sleep(60)
                
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        
        metrics = {}
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics.update({
                "system_cpu_percent": cpu_percent,
                "system_memory_percent": memory.percent,
                "system_memory_available_gb": memory.available / (1024**3),
                "system_memory_used_gb": memory.used / (1024**3)
            })
            
            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    metrics.update({
                        "gpu_utilization": gpu.load * 100,
                        "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                        "gpu_memory_total_gb": gpu.memoryTotal / 1024,
                        "gpu_temperature": gpu.temperature
                    })
            except ImportError:
                pass  # GPU monitoring not available
                
        except Exception as e:
            self.logger.warning(f"Error collecting system metrics: {e}")
            
        return metrics
        
    async def _collect_ray_cluster_metrics(self) -> Dict[str, Any]:
        """Collect Ray cluster metrics."""
        
        metrics = {}
        
        try:
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                cluster_state = ray.state.cluster_resources()
                
                metrics.update({
                    "ray_cluster_cpus": cluster_resources.get("CPU", 0),
                    "ray_cluster_gpus": cluster_resources.get("GPU", 0),
                    "ray_cluster_memory_gb": cluster_resources.get("memory", 0) / (1024**3),
                    "ray_cluster_nodes": len(ray.nodes()),
                    "ray_active_tasks": len(ray.state.tasks()),
                    "ray_active_actors": len(ray.state.actors())
                })
                
        except Exception as e:
            self.logger.warning(f"Error collecting Ray cluster metrics: {e}")
            
        return metrics
        
    def _get_standard_metric_names(self) -> List[str]:
        """Get list of standard metric names."""
        return [
            "policy_loss", "value_loss", "reward_mean", "reward_std",
            "episode_length_mean", "kl_divergence", "entropy",
            "samples_per_second", "steps_per_second", "tokens_per_second",
            "gpu_utilization", "memory_usage_gb", "cluster_nodes_active",
            "gradient_sync_time", "communication_overhead", "verl_algorithm",
            "inference_time", "model_loading_time"
        ]
        
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules for common issues."""
        
        default_rules = [
            AlertRule(
                rule_id="high_policy_loss",
                name="High Policy Loss",
                metric_name="policy_loss",
                condition="greater_than",
                threshold=5.0,
                severity=MetricSeverity.WARNING
            ),
            AlertRule(
                rule_id="low_reward_mean",
                name="Low Average Reward",
                metric_name="reward_mean",
                condition="less_than", 
                threshold=-10.0,
                severity=MetricSeverity.WARNING
            ),
            AlertRule(
                rule_id="high_gpu_utilization",
                name="High GPU Utilization",
                metric_name="gpu_utilization",
                condition="greater_than",
                threshold=95.0,
                severity=MetricSeverity.WARNING
            ),
            AlertRule(
                rule_id="low_throughput",
                name="Low Training Throughput",
                metric_name="samples_per_second",
                condition="less_than",
                threshold=10.0,
                severity=MetricSeverity.WARNING
            ),
            AlertRule(
                rule_id="high_kl_divergence",
                name="High KL Divergence", 
                metric_name="kl_divergence",
                condition="greater_than",
                threshold=0.1,
                severity=MetricSeverity.ERROR
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
            
    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against latest metrics."""
        
        if not self.metrics_history:
            return
            
        latest_metrics = self.metrics_history[-1]
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
                
            # Get metric value
            metric_value = getattr(latest_metrics, rule.metric_name, None)
            if metric_value is None:
                metric_value = latest_metrics.custom_metrics.get(rule.metric_name)
                
            if metric_value is None:
                continue
                
            # Evaluate condition
            violated = False
            if rule.condition == "greater_than" and metric_value > rule.threshold:
                violated = True
            elif rule.condition == "less_than" and metric_value < rule.threshold:
                violated = True
            elif rule.condition == "equals" and metric_value == rule.threshold:
                violated = True
            elif rule.condition == "not_equals" and metric_value != rule.threshold:
                violated = True
                
            if violated:
                self.alert_violations[rule_id] += 1
                
                # Trigger alert if consecutive violations threshold met
                if self.alert_violations[rule_id] >= rule.consecutive_violations:
                    await self._trigger_alert(rule, metric_value, latest_metrics)
                    
            else:
                # Reset violation count
                self.alert_violations[rule_id] = 0
                
    async def _trigger_alert(
        self, 
        rule: AlertRule, 
        metric_value: float, 
        metrics: TrainingMetrics
    ) -> None:
        """Trigger an alert."""
        
        # Check cooldown
        last_alert_time = self.last_alert_times.get(rule.rule_id, 0)
        if time.time() - last_alert_time < rule.cooldown_seconds:
            return
            
        alert_data = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "metric_value": metric_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "severity": rule.severity.value,
            "training_step": metrics.training_step,
            "timestamp": metrics.timestamp,
            "phase": metrics.phase.value
        }
        
        # Log alert
        self.logger.warning(
            f"ALERT: {rule.name}",
            **alert_data
        )
        
        # Run alert callbacks
        await self._run_alert_callbacks(alert_data)
        
        # Update last alert time
        self.last_alert_times[rule.rule_id] = time.time()
        
    # Analysis methods
    def _analyze_training_efficiency(self) -> Dict[str, Any]:
        """Analyze training efficiency metrics."""
        
        if len(self.metrics_history) < 10:
            return {"error": "Insufficient data"}
            
        recent_metrics = self.metrics_history[-50:]  # Last 50 data points
        
        # Calculate efficiency trends
        throughput_trend = [m.samples_per_second for m in recent_metrics]
        loss_trend = [m.policy_loss for m in recent_metrics if m.policy_loss > 0]
        reward_trend = [m.reward_mean for m in recent_metrics]
        
        return {
            "avg_throughput": sum(throughput_trend) / len(throughput_trend),
            "throughput_stability": self._calculate_stability(throughput_trend),
            "loss_convergence": self._calculate_convergence(loss_trend),
            "reward_improvement": self._calculate_improvement(reward_trend),
            "efficiency_score": self._calculate_efficiency_score(recent_metrics)
        }
        
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        
        recent_metrics = self.metrics_history[-50:]
        
        gpu_utils = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization > 0]
        memory_usage = [m.memory_usage_gb for m in recent_metrics if m.memory_usage_gb > 0]
        
        return {
            "avg_gpu_utilization": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
            "gpu_utilization_efficiency": "high" if sum(gpu_utils) / len(gpu_utils) > 80 else "low",
            "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "resource_bottlenecks": self._identify_resource_bottlenecks(recent_metrics)
        }
        
    def _analyze_distributed_performance(self) -> Dict[str, Any]:
        """Analyze distributed training performance."""
        
        recent_metrics = self.metrics_history[-20:]
        
        sync_times = [m.gradient_sync_time for m in recent_metrics if m.gradient_sync_time > 0]
        comm_overhead = [m.communication_overhead for m in recent_metrics if m.communication_overhead > 0]
        
        return {
            "avg_gradient_sync_time": sum(sync_times) / len(sync_times) if sync_times else 0,
            "avg_communication_overhead": sum(comm_overhead) / len(comm_overhead) if comm_overhead else 0,
            "distributed_efficiency": self._calculate_distributed_efficiency(recent_metrics),
            "scaling_effectiveness": self._analyze_scaling_effectiveness(recent_metrics)
        }
        
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        
        if len(self.metrics_history) < 20:
            return {"error": "Insufficient data"}
            
        loss_values = [m.policy_loss for m in self.metrics_history if m.policy_loss > 0]
        reward_values = [m.reward_mean for m in self.metrics_history]
        
        return {
            "loss_trend": "decreasing" if self._is_decreasing(loss_values[-10:]) else "stable",
            "reward_trend": "increasing" if self._is_increasing(reward_values[-10:]) else "stable",
            "convergence_stability": self._calculate_convergence_stability(self.metrics_history[-20:]),
            "estimated_convergence_steps": self._estimate_convergence_steps(loss_values)
        }
        
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in training metrics."""
        
        anomalies = []
        
        if len(self.metrics_history) < 10:
            return anomalies
            
        # Simple anomaly detection based on standard deviations
        recent_metrics = self.metrics_history[-20:]
        
        # Check reward anomalies
        rewards = [m.reward_mean for m in recent_metrics]
        reward_mean = sum(rewards) / len(rewards)
        reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        
        for metric in recent_metrics[-5:]:  # Check last 5 metrics
            if abs(metric.reward_mean - reward_mean) > 3 * reward_std:
                anomalies.append({
                    "type": "reward_anomaly",
                    "step": metric.training_step,
                    "value": metric.reward_mean,
                    "expected_range": [reward_mean - 2*reward_std, reward_mean + 2*reward_std]
                })
                
        return anomalies
        
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        
        if len(self.metrics_history) < 10:
            return ["Collect more training data for detailed recommendations"]
            
        recent_metrics = self.metrics_history[-20:]
        
        # Throughput recommendations
        avg_throughput = sum(m.samples_per_second for m in recent_metrics) / len(recent_metrics)
        if avg_throughput < 50:
            recommendations.append("Consider increasing batch size or optimizing data loading pipeline")
            
        # GPU utilization recommendations
        gpu_utils = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization > 0]
        if gpu_utils and sum(gpu_utils) / len(gpu_utils) < 60:
            recommendations.append("GPU utilization is low - consider increasing model size or batch size")
            
        # Distributed training recommendations
        sync_times = [m.gradient_sync_time for m in recent_metrics if m.gradient_sync_time > 0]
        if sync_times and sum(sync_times) / len(sync_times) > 0.5:
            recommendations.append("High gradient synchronization time - check network bandwidth and reduce communication frequency")
            
        # Convergence recommendations
        loss_values = [m.policy_loss for m in recent_metrics if m.policy_loss > 0]
        if len(loss_values) >= 10 and not self._is_decreasing(loss_values[-10:]):
            recommendations.append("Policy loss not decreasing - consider adjusting learning rate or reward function")
            
        return recommendations[:5]  # Return top 5 recommendations
        
    # Utility methods for analysis
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score (lower variance = higher stability)."""
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return max(0, 1 - variance / (mean_val + 1e-8))
        
    def _calculate_convergence(self, values: List[float]) -> float:
        """Calculate convergence score (higher = more converged).""" 
        if len(values) < 5:
            return 0.0
        recent_std = (sum((v - sum(values[-5:]) / 5) ** 2 for v in values[-5:]) / 5) ** 0.5
        overall_mean = sum(values) / len(values)
        return max(0, 1 - recent_std / (abs(overall_mean) + 1e-8))
        
    def _calculate_improvement(self, values: List[float]) -> float:
        """Calculate improvement rate."""
        if len(values) < 10:
            return 0.0
        early_avg = sum(values[:5]) / 5
        recent_avg = sum(values[-5:]) / 5
        return (recent_avg - early_avg) / (abs(early_avg) + 1e-8)
        
    def _calculate_efficiency_score(self, metrics: List[TrainingMetrics]) -> float:
        """Calculate overall efficiency score."""
        if not metrics:
            return 0.0
        
        # Combine multiple efficiency indicators
        throughput_score = min(1.0, sum(m.samples_per_second for m in metrics) / (len(metrics) * 100))
        gpu_score = sum(m.gpu_utilization for m in metrics if m.gpu_utilization > 0) / (len(metrics) * 100)
        
        return (throughput_score + gpu_score) / 2
        
    def _identify_resource_bottlenecks(self, metrics: List[TrainingMetrics]) -> List[str]:
        """Identify resource bottlenecks."""
        bottlenecks = []
        
        avg_gpu = sum(m.gpu_utilization for m in metrics if m.gpu_utilization > 0) / len(metrics)
        avg_memory = sum(m.memory_usage_gb for m in metrics if m.memory_usage_gb > 0) / len(metrics)
        
        if avg_gpu > 95:
            bottlenecks.append("GPU utilization at capacity")
        if avg_memory > 0.9:  # Assuming 1GB is the limit for this example
            bottlenecks.append("Memory usage high")
            
        return bottlenecks
        
    def _calculate_distributed_efficiency(self, metrics: List[TrainingMetrics]) -> float:
        """Calculate distributed training efficiency."""
        if not metrics:
            return 0.0
            
        sync_times = [m.gradient_sync_time for m in metrics if m.gradient_sync_time > 0]
        if not sync_times:
            return 1.0
            
        avg_sync_time = sum(sync_times) / len(sync_times)
        # Efficiency decreases with higher sync times
        return max(0, 1 - avg_sync_time / 1.0)  # Normalize by 1 second
        
    def _analyze_scaling_effectiveness(self, metrics: List[TrainingMetrics]) -> str:
        """Analyze how effectively the system scales with more nodes."""
        node_counts = [m.cluster_nodes_active for m in metrics]
        throughputs = [m.samples_per_second for m in metrics]
        
        if len(set(node_counts)) <= 1:
            return "single_node"
            
        # Simple analysis - would be more sophisticated in practice
        return "scaling_well" if max(throughputs) > min(throughputs) * 1.5 else "scaling_poorly"
        
    def _is_decreasing(self, values: List[float]) -> bool:
        """Check if values show a decreasing trend."""
        if len(values) < 3:
            return False
        return sum(values[:len(values)//2]) > sum(values[len(values)//2:])
        
    def _is_increasing(self, values: List[float]) -> bool:
        """Check if values show an increasing trend."""
        if len(values) < 3:
            return False
        return sum(values[:len(values)//2]) < sum(values[len(values)//2:])
        
    def _calculate_convergence_stability(self, metrics: List[TrainingMetrics]) -> float:
        """Calculate convergence stability score."""
        if len(metrics) < 5:
            return 0.0
            
        rewards = [m.reward_mean for m in metrics]
        losses = [m.policy_loss for m in metrics if m.policy_loss > 0]
        
        reward_stability = self._calculate_stability(rewards)
        loss_stability = self._calculate_stability(losses) if losses else 0
        
        return (reward_stability + loss_stability) / 2
        
    def _estimate_convergence_steps(self, loss_values: List[float]) -> Optional[int]:
        """Estimate steps until convergence."""
        if len(loss_values) < 10:
            return None
            
        # Simple linear extrapolation - would use more sophisticated methods in practice
        recent_trend = loss_values[-5:]
        if self._is_decreasing(recent_trend):
            return len(loss_values) + 100  # Rough estimate
        return None
        
    # Integration and callback methods
    async def _setup_ray_monitoring(self) -> None:
        """Setup Ray cluster monitoring integration."""
        try:
            if not ray.is_initialized():
                self.logger.warning("Ray not initialized - cannot setup Ray monitoring")
                return
                
            self.ray_client = ray
            self.logger.info("Ray monitoring integration setup complete")
            
        except Exception as e:
            self.logger.error(f"Failed to setup Ray monitoring: {e}")
            
    async def _setup_external_integrations(self) -> None:
        """Setup external monitoring integrations."""
        
        # TensorBoard integration
        if self.config.tensorboard_integration:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self.config.metrics_export_path) / "tensorboard"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir)
                self.logger.info("TensorBoard integration enabled")
            except ImportError:
                self.logger.warning("TensorBoard not available")
                
        # Weights & Biases integration
        if self.config.wandb_integration:
            try:
                import wandb
                # wandb.init() would be called here with proper config
                self.logger.info("W&B integration enabled")
            except ImportError:
                self.logger.warning("W&B not available")
                
    def _get_active_integrations(self) -> List[str]:
        """Get list of active integrations."""
        integrations = []
        
        if self.ray_client:
            integrations.append("ray")
        if self.tensorboard_writer:
            integrations.append("tensorboard")
        if self.wandb_run:
            integrations.append("wandb")
            
        return integrations
        
    async def _export_metrics(self, metrics: TrainingMetrics) -> None:
        """Export metrics to external systems."""
        
        # Export to file
        if self.config.export_metrics_to_file:
            await self._export_to_file(metrics)
            
        # Export to TensorBoard
        if self.tensorboard_writer:
            await self._export_to_tensorboard(metrics)
            
        # Export to W&B
        if self.wandb_run:
            await self._export_to_wandb(metrics)
            
    async def _export_to_file(self, metrics: TrainingMetrics) -> None:
        """Export metrics to file."""
        try:
            export_path = Path(self.config.metrics_export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Export as JSON lines
            metrics_file = export_path / "training_metrics.jsonl"
            with open(metrics_file, 'a') as f:
                json.dump(metrics.__dict__, f, default=str)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics to file: {e}")
            
    async def _export_to_tensorboard(self, metrics: TrainingMetrics) -> None:
        """Export metrics to TensorBoard."""
        try:
            step = metrics.training_step
            
            # Log scalar metrics
            self.tensorboard_writer.add_scalar('reward/mean', metrics.reward_mean, step)
            self.tensorboard_writer.add_scalar('loss/policy', metrics.policy_loss, step)
            self.tensorboard_writer.add_scalar('loss/value', metrics.value_loss, step)
            self.tensorboard_writer.add_scalar('performance/throughput', metrics.samples_per_second, step)
            self.tensorboard_writer.add_scalar('performance/gpu_utilization', metrics.gpu_utilization, step)
            
            # Custom metrics
            for name, value in metrics.custom_metrics.items():
                self.tensorboard_writer.add_scalar(f'custom/{name}', value, step)
                
            self.tensorboard_writer.flush()
            
        except Exception as e:
            self.logger.error(f"Failed to export to TensorBoard: {e}")
            
    async def _export_to_wandb(self, metrics: TrainingMetrics) -> None:
        """Export metrics to Weights & Biases."""
        try:
            import wandb
            
            log_data = {
                'step': metrics.training_step,
                'reward_mean': metrics.reward_mean,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'throughput': metrics.samples_per_second,
                'gpu_utilization': metrics.gpu_utilization
            }
            
            # Add custom metrics
            log_data.update(metrics.custom_metrics)
            
            wandb.log(log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to export to W&B: {e}")
            
    async def _run_metric_callbacks(self, metrics: TrainingMetrics) -> None:
        """Run registered metric callbacks."""
        for callback in self.metric_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                self.logger.warning(f"Metric callback failed: {e}")
                
    async def _run_alert_callbacks(self, alert_data: Dict[str, Any]) -> None:
        """Run registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                self.logger.warning(f"Alert callback failed: {e}")
                
    # Public callback registration methods
    def register_metric_callback(self, callback: Callable) -> None:
        """Register a callback for metric events."""
        self.metric_callbacks.append(callback)
        
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for alert events."""
        self.alert_callbacks.append(callback)
        
    # Cleanup methods
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and cleanup resources."""
        
        self.logger.info("Stopping VERL training monitoring")
        
        self.monitoring_active = False
        
        # Close external integrations
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except:
                pass
                
        # Generate final summary
        final_summary = await self.get_training_summary()
        
        # Export final metrics
        if self.config.export_metrics_to_file and self.metrics_history:
            final_export_path = Path(self.config.metrics_export_path) / "final_training_summary.json"
            with open(final_export_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
                
        training_duration = (time.time() - self.training_start_time) if self.training_start_time else 0
        
        self.logger.info(
            "VERL training monitoring stopped",
            total_duration=training_duration,
            total_metrics=len(self.metrics_history),
            final_summary=final_summary.get("training_status", {})
        )
        
        return {
            "success": True,
            "monitoring_stopped_at": time.time(),
            "total_duration": training_duration,
            "total_metrics_collected": len(self.metrics_history),
            "final_summary": final_summary
        }