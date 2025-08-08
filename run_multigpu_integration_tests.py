#!/usr/bin/env python3
"""
Multi-GPU Phase 3 Integration Test Runner

Optimized test runner for multi-GPU setups with distributed VERL training,
parallel evaluation, and comprehensive monitoring.
"""

import asyncio
import sys
import time
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import torch
import ray

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import Phase 3 components
from coding_framework.verl_integration.verl_coordinator import VERLCoordinator
from coding_framework.verl_integration.verl_config import VERLDistributedConfig
from coding_framework.verl_integration.multi_agent_trainer import MultiAgentVERLTrainer

from coding_framework.evaluation.benchmark_manager import BenchmarkManager, BenchmarkManagerConfig
from coding_framework.evaluation.verl_evaluation_bridge import VERLEvaluationBridge, VERLEvaluationConfig

from coding_framework.deployment.deployment_orchestrator import DeploymentOrchestrator, OrchestratorConfig
from coding_framework.deployment.deployment_manager import DeploymentTarget

from coding_framework.monitoring.verl_training_monitor import (
    VERLTrainingMonitor,
    VERLTrainingMonitorConfig, 
    TrainingPhase
)

from coding_framework.agents.code_generator_agent import CodeGeneratorAgent
from coding_framework.agents.agent_config import CodeGeneratorConfig


class MultiGPUPhase3Tester:
    """Multi-GPU optimized integration tester for Phase 3."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        self.gpu_count = torch.cuda.device_count()
        self.ray_initialized = ray.is_initialized()
        
        # Load configurations
        self.load_configs()
        
    def load_configs(self):
        """Load multi-GPU configurations."""
        config_path = Path("configs")
        
        # Load VERL config
        with open(config_path / "verl_multigpu_config.json") as f:
            self.verl_config_data = json.load(f)
            
        # Load evaluation config
        with open(config_path / "evaluation_multigpu_config.json") as f:
            self.eval_config_data = json.load(f)
            
        # Load deployment config
        with open(config_path / "deployment_multigpu_config.json") as f:
            self.deployment_config_data = json.load(f)
            
        # Load monitoring config
        with open(config_path / "monitoring_multigpu_config.json") as f:
            self.monitoring_config_data = json.load(f)
            
    def log_test_progress(self, test_name: str, status: str, details: str = ""):
        """Log test progress with GPU info."""
        elapsed = time.time() - self.start_time
        gpu_info = f"[{self.gpu_count}xGPU]" if self.gpu_count > 1 else "[1xGPU]"
        print(f"[{elapsed:.1f}s] {gpu_info} {test_name}: {status}")
        if details:
            print(f"    {details}")
            
    def record_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any] = None):
        """Record test result with GPU utilization info."""
        gpu_memory_info = {}
        if torch.cuda.is_available():
            for i in range(self.gpu_count):
                memory_used = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                gpu_memory_info[f"gpu_{i}"] = {
                    "memory_used_gb": round(memory_used, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "utilization_pct": round((memory_used / memory_total) * 100, 1)
                }
                
        self.test_results[test_name] = {
            "success": success,
            "duration": duration,
            "details": details or {},
            "gpu_info": {
                "gpu_count": self.gpu_count,
                "memory_info": gpu_memory_info
            }
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        self.log_test_progress(test_name, f"{status} ({duration:.2f}s)")
        
    async def test_multigpu_verl_integration(self) -> bool:
        """Test VERL integration with multi-GPU setup."""
        
        test_start = time.time()
        self.log_test_progress("Multi-GPU VERL Integration", "STARTING")
        
        try:
            # Create VERL config optimized for available GPUs
            verl_config = VERLDistributedConfig(
                algorithm=self.verl_config_data["algorithm"],
                num_gpus=min(self.gpu_count, self.verl_config_data["num_gpus"]),
                strategy=self.verl_config_data["strategy"],
                ray_config=self.verl_config_data["ray_config"]
            )
            
            # Adjust worker count based on available GPUs
            verl_config.ray_config["num_workers"] = min(self.gpu_count, 4)
            
            coordinator = VERLCoordinator(verl_config)
            
            # Test initialization with GPU distribution
            self.log_test_progress("VERL Coordinator", "Initializing with GPU distribution...")
            init_result = await coordinator.initialize()
            
            if not init_result.get("success", False):
                raise Exception(f"VERL initialization failed: {init_result}")
                
            # Verify Ray cluster resources
            if self.ray_initialized:
                cluster_resources = ray.cluster_resources()
                available_gpus = cluster_resources.get("GPU", 0)
                self.log_test_progress("Ray Cluster", f"Available GPUs: {available_gpus}")
                
            # Test multi-agent trainer with GPU distribution
            trainer = MultiAgentVERLTrainer(verl_config)
            
            # Create multiple test agents for distributed training
            agents = []
            for i in range(min(2, self.gpu_count)):  # Max 2 agents for testing
                config = CodeGeneratorConfig()
                config.model_name = f"test_agent_{i}"
                agent = CodeGeneratorAgent(config)
                agents.append(agent)
                
            setup_result = await trainer.setup_agents(agents)
            if not setup_result.get("success", False):
                raise Exception(f"Multi-agent setup failed: {setup_result}")
                
            # Test distributed training simulation
            training_config = self.verl_config_data["training_config"]
            training_config["batch_size"] = training_config["batch_size"] // max(1, self.gpu_count // 2)  # Adjust for available GPUs
            
            training_result = await coordinator.start_training(training_config)
            if not training_result.get("success", False):
                raise Exception(f"Distributed training failed: {training_result}")
                
            # Test checkpoint with multi-GPU state
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = Path(temp_dir) / "multigpu_checkpoint"
                save_result = await coordinator.save_checkpoint(str(checkpoint_path), step=50)
                
                if not save_result.get("success", False):
                    raise Exception(f"Multi-GPU checkpoint save failed: {save_result}")
                    
            duration = time.time() - test_start
            self.record_test_result("Multi-GPU VERL Integration", True, duration, {
                "gpus_used": verl_config.num_gpus,
                "workers_configured": verl_config.ray_config["num_workers"],
                "agents_trained": len(agents),
                "checkpoint_saved": True
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Multi-GPU VERL Integration", False, duration, {"error": str(e)})
            return False
            
    async def test_parallel_evaluation_framework(self) -> bool:
        """Test evaluation framework with parallel GPU execution."""
        
        test_start = time.time()
        self.log_test_progress("Parallel Evaluation Framework", "STARTING")
        
        try:
            # Setup parallel evaluation configuration
            eval_config = BenchmarkManagerConfig(
                results_dir="./test_results/evaluation",
                parallel_execution=True,
                max_concurrent_evaluations=min(self.gpu_count, 3),
                save_detailed_results=True
            )
            
            benchmark_manager = BenchmarkManager(eval_config)
            
            # Create multiple test agents for parallel evaluation
            test_agents = []
            for i in range(min(2, self.gpu_count)):
                config = CodeGeneratorConfig()
                config.model_name = f"eval_agent_{i}"
                agent = CodeGeneratorAgent(config)
                test_agents.append(agent)
                
            # Test parallel evaluation on multiple benchmarks
            benchmark_configs = self.eval_config_data["benchmarks"]
            
            # Run evaluations in parallel for each agent
            evaluation_tasks = []
            for i, agent in enumerate(test_agents):
                task = benchmark_manager.run_comprehensive_evaluation(
                    agent=agent,
                    benchmarks=["humaneval"],  # Focus on HumanEval for multi-GPU testing
                    save_results=True
                )
                evaluation_tasks.append(task)
                
            # Wait for all parallel evaluations
            self.log_test_progress("Parallel Evaluation", f"Running {len(evaluation_tasks)} parallel evaluations...")
            evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Verify results
            successful_evaluations = 0
            for i, result in enumerate(evaluation_results):
                if isinstance(result, Exception):
                    self.log_test_progress("Evaluation Error", f"Agent {i}: {result}")
                else:
                    if "agent_id" in result and result.get("benchmarks"):
                        successful_evaluations += 1
                        
            if successful_evaluations == 0:
                raise Exception("No evaluations completed successfully")
                
            # Test VERL evaluation bridge with GPU awareness
            bridge_config = VERLEvaluationConfig(
                evaluation_frequency=10,
                benchmarks=["humaneval"],
                quick_eval_problems=benchmark_configs["humaneval"]["max_samples"]
            )
            
            eval_bridge = VERLEvaluationBridge(bridge_config)
            
            # Test evaluation during training with first agent
            bridge_result = await eval_bridge.evaluate_during_training(
                agent=test_agents[0],
                step=100,
                full_evaluation=True
            )
            
            if not bridge_result or bridge_result.get("step") != 100:
                raise Exception(f"VERL evaluation bridge failed: {bridge_result}")
                
            duration = time.time() - test_start
            self.record_test_result("Parallel Evaluation Framework", True, duration, {
                "parallel_evaluations": len(test_agents),
                "successful_evaluations": successful_evaluations,
                "max_concurrent": eval_config.max_concurrent_evaluations,
                "verl_bridge_tested": True
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Parallel Evaluation Framework", False, duration, {"error": str(e)})
            return False
            
    async def test_distributed_deployment_infrastructure(self) -> bool:
        """Test deployment infrastructure with GPU-aware resource allocation."""
        
        test_start = time.time()
        self.log_test_progress("Distributed Deployment Infrastructure", "STARTING")
        
        try:
            # Create temporary model file
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
                model_path = temp_file.name
                temp_file.write(b"multigpu_trained_model_data")
                
            try:
                # Setup orchestrator with GPU-aware configuration
                orchestrator_config = OrchestratorConfig(
                    enable_automated_rollback=True,
                    enable_deployment_monitoring=False,  # Simplified for testing
                    max_deployment_time=300,  # 5 minutes for testing
                    enable_parallel_deployments=True,
                    max_concurrent_deployments=min(self.gpu_count, 2)
                )
                
                orchestrator = DeploymentOrchestrator(orchestrator_config)
                
                # Configure deployment with GPU resources
                from coding_framework.deployment.deployment_manager import DeploymentConfig
                deployment_config = DeploymentConfig(
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT,
                    deployment_strategy="blue_green",
                    replicas=min(self.gpu_count, self.deployment_config_data["replicas"]),
                    resource_requirements={
                        "gpu": min(1, self.gpu_count),
                        "cpu": 2,
                        "memory": "4Gi"
                    }
                )
                
                # Configure model server with GPU support
                from coding_framework.deployment.model_server import ModelServerConfig
                server_config = ModelServerConfig(
                    model_path=model_path,
                    port=8110,  # Unique port for multi-GPU testing
                    workers=min(self.gpu_count, 2),
                    gpu_limit=min(1, self.gpu_count),
                    enable_batching=True,
                    batch_size=16 * min(self.gpu_count, 2)  # Scale batch size
                )
                
                # Test deployment workflow with GPU resources
                self.log_test_progress("Deployment Orchestrator", f"Deploying with {deployment_config.replicas} replicas...")
                
                deployment_result = await orchestrator.deploy_model_full_workflow(
                    workflow_id="multigpu_test_deployment",
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT,
                    deployment_config=deployment_config,
                    server_config=server_config
                )
                
                if not deployment_result.get("success", False):
                    raise Exception(f"Multi-GPU deployment failed: {deployment_result}")
                    
                # Test parallel model server instances
                if deployment_result.get("deployment_endpoints"):
                    self.log_test_progress("Model Servers", f"Deployed {len(deployment_result['deployment_endpoints'])} endpoints")
                    
                # Test deployment health with GPU utilization
                health_result = await orchestrator.get_deployment_health("multigpu_test_deployment")
                if "workflow_id" not in health_result:
                    raise Exception(f"Health check failed: {health_result}")
                    
                duration = time.time() - test_start
                self.record_test_result("Distributed Deployment Infrastructure", True, duration, {
                    "replicas_deployed": deployment_config.replicas,
                    "gpu_resources_allocated": server_config.gpu_limit,
                    "endpoints_created": len(deployment_result.get("deployment_endpoints", [])),
                    "health_check_passed": True
                })
                return True
                
            finally:
                # Cleanup
                Path(model_path).unlink(missing_ok=True)
                
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Distributed Deployment Infrastructure", False, duration, {"error": str(e)})
            return False
            
    async def test_multigpu_monitoring_observability(self) -> bool:
        """Test monitoring with multi-GPU resource tracking."""
        
        test_start = time.time()
        self.log_test_progress("Multi-GPU Monitoring & Observability", "STARTING")
        
        try:
            # Setup monitoring with GPU-specific configuration
            monitor_config = VERLTrainingMonitorConfig(
                metrics_collection_interval=5,
                enable_alerts=True,
                export_metrics_to_file=True,
                metrics_export_path="./test_results/monitoring",
                profile_gpu_usage=True,
                monitor_distributed_metrics=True
            )
            
            monitor = VERLTrainingMonitor(monitor_config)
            
            # Test monitor startup with GPU detection
            start_result = await monitor.start_monitoring()
            if not start_result.get("success", False):
                raise Exception(f"Multi-GPU monitor startup failed: {start_result}")
                
            # Simulate multi-GPU training metrics
            gpu_metrics_samples = []
            for step in range(5):
                # Generate metrics for each GPU
                gpu_specific_metrics = {}
                total_gpu_util = 0
                total_memory = 0
                
                for gpu_id in range(self.gpu_count):
                    gpu_util = 60.0 + step * 8.0 + gpu_id * 5.0  # Varying utilization per GPU
                    memory_usage = 4.0 + step * 0.5 + gpu_id * 0.2  # GB
                    
                    gpu_specific_metrics[f"gpu_{gpu_id}_utilization"] = min(gpu_util, 95.0)
                    gpu_specific_metrics[f"gpu_{gpu_id}_memory_gb"] = memory_usage
                    
                    total_gpu_util += gpu_util
                    total_memory += memory_usage
                    
                # Aggregate metrics
                step_metrics = {
                    "policy_loss": max(3.0 - step * 0.4, 0.5),
                    "value_loss": max(2.0 - step * 0.3, 0.3),
                    "reward_mean": -8.0 + step * 2.5,
                    "samples_per_second": 80.0 * self.gpu_count + step * 20.0,  # Scale with GPUs
                    "gpu_utilization": total_gpu_util / self.gpu_count,  # Average
                    "memory_usage_gb": total_memory,  # Total across GPUs
                    "cluster_nodes_active": min(self.gpu_count, 4),
                    "gradient_sync_time": 0.1 + (self.gpu_count - 1) * 0.05,  # Increases with more GPUs
                    **gpu_specific_metrics
                }
                
                await monitor.record_training_metrics(
                    metrics=step_metrics,
                    training_step=step * 50,
                    phase=TrainingPhase.POLICY_TRAINING
                )
                
                gpu_metrics_samples.append(step_metrics)
                
            # Test multi-GPU evaluation results integration
            eval_results = {
                "benchmarks": {
                    "humaneval": {
                        "pass_at_1": 0.48,
                        "pass_at_10": 0.73,
                        "solved_problems": 79,
                        "total_problems": 164,
                        "gpu_evaluation_time": 45.2,
                        "parallel_execution": True
                    }
                },
                "aggregate": {
                    "weighted_pass_at_1": 0.48,
                    "overall_pass_rate": 0.48,
                    "total_evaluation_gpus": self.gpu_count
                }
            }
            
            await monitor.record_evaluation_results(
                evaluation_results=eval_results,
                training_step=250
            )
            
            # Test training summary with GPU-specific insights
            summary = await monitor.get_training_summary()
            if not summary.get("training_status", {}).get("monitoring_active"):
                raise Exception("Training summary shows monitoring inactive")
                
            # Test performance analysis with multi-GPU considerations
            analysis = await monitor.get_performance_analysis()
            if "distributed_performance" not in analysis:
                raise Exception("Performance analysis missing distributed metrics")
                
            # Verify GPU-specific metrics are tracked
            latest_metrics = monitor.metrics_history[-1] if monitor.metrics_history else None
            gpu_metrics_found = 0
            
            if latest_metrics:
                for metric_name in latest_metrics.custom_metrics:
                    if "gpu_" in metric_name:
                        gpu_metrics_found += 1
                        
            if gpu_metrics_found < self.gpu_count:
                raise Exception(f"Expected {self.gpu_count} GPU metrics, found {gpu_metrics_found}")
                
            # Test monitoring shutdown with multi-GPU summary
            stop_result = await monitor.stop_monitoring()
            if not stop_result.get("success", False):
                raise Exception(f"Multi-GPU monitor shutdown failed: {stop_result}")
                
            duration = time.time() - test_start
            self.record_test_result("Multi-GPU Monitoring & Observability", True, duration, {
                "gpus_monitored": self.gpu_count,
                "gpu_specific_metrics": gpu_metrics_found,
                "distributed_analysis": True,
                "total_metrics": stop_result.get("total_metrics_collected", 0),
                "avg_gpu_utilization": sum(m.get("gpu_utilization", 0) for m in gpu_metrics_samples) / len(gpu_metrics_samples)
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Multi-GPU Monitoring & Observability", False, duration, {"error": str(e)})
            return False
            
    async def test_multigpu_end_to_end_integration(self) -> bool:
        """Test complete multi-GPU end-to-end workflow."""
        
        test_start = time.time()
        self.log_test_progress("Multi-GPU End-to-End Integration", "STARTING")
        
        try:
            # Phase 1: Multi-GPU VERL training setup
            verl_config = VERLDistributedConfig(
                algorithm="ppo",
                num_gpus=min(self.gpu_count, 4),
                strategy="fsdp2"
            )
            
            coordinator = VERLCoordinator(verl_config)
            await coordinator.initialize()
            
            # Phase 2: Distributed monitoring setup
            monitor = VERLTrainingMonitor()
            await monitor.start_monitoring(verl_config)
            
            # Phase 3: Simulate distributed training
            self.log_test_progress("E2E Training", f"Simulating training across {verl_config.num_gpus} GPUs...")
            
            for step in range(3):
                # Multi-GPU training metrics
                distributed_metrics = {
                    "policy_loss": max(2.8 - step * 0.6, 0.4),
                    "reward_mean": -6.0 + step * 2.8,
                    "samples_per_second": 120.0 * verl_config.num_gpus + step * 30.0,
                    "gpu_utilization": min(70.0 + step * 10.0, 92.0),
                    "memory_usage_gb": 5.5 * verl_config.num_gpus,
                    "cluster_nodes_active": verl_config.num_gpus,
                    "gradient_sync_time": 0.08 + verl_config.num_gpus * 0.02,
                    "communication_overhead": verl_config.num_gpus * 0.015
                }
                
                await monitor.record_training_metrics(
                    metrics=distributed_metrics,
                    training_step=step * 100,
                    phase=TrainingPhase.POLICY_TRAINING
                )
                
            # Phase 4: Parallel evaluation
            test_agent = CodeGeneratorAgent(CodeGeneratorConfig())
            eval_bridge = VERLEvaluationBridge()
            
            eval_result = await eval_bridge.evaluate_during_training(
                agent=test_agent,
                step=300,
                full_evaluation=False
            )
            
            if eval_result.get("results"):
                await monitor.record_evaluation_results(
                    evaluation_results=eval_result["results"],
                    training_step=300
                )
                
            # Phase 5: Multi-GPU deployment
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
                model_path = temp_file.name
                temp_file.write(b"e2e_multigpu_model")
                
            try:
                orchestrator = DeploymentOrchestrator()
                
                # GPU-aware deployment configuration
                from coding_framework.deployment.deployment_manager import DeploymentConfig
                deployment_config = DeploymentConfig(
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT,
                    replicas=min(self.gpu_count, 2)
                )
                
                from coding_framework.deployment.model_server import ModelServerConfig
                server_config = ModelServerConfig(
                    model_path=model_path,
                    port=8115,
                    gpu_limit=min(1, self.gpu_count)
                )
                
                deploy_result = await orchestrator.deploy_model_full_workflow(
                    workflow_id="multigpu_e2e_integration",
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT,
                    deployment_config=deployment_config,
                    server_config=server_config
                )
                
                if not deploy_result.get("success"):
                    raise Exception(f"E2E multi-GPU deployment failed: {deploy_result}")
                    
                # Phase 6: Final monitoring summary with GPU insights
                final_summary = await monitor.get_training_summary()
                
                # Verify distributed performance metrics
                perf_summary = final_summary.get("performance_summary", {})
                distributed_section = final_summary.get("distributed_training", {})
                
                if not distributed_section:
                    raise Exception("Missing distributed training metrics in final summary")
                    
                await monitor.stop_monitoring()
                
                duration = time.time() - test_start
                self.record_test_result("Multi-GPU End-to-End Integration", True, duration, {
                    "gpus_utilized": verl_config.num_gpus,
                    "training_simulation": True,
                    "evaluation_integration": True,
                    "deployment_success": True,
                    "monitoring_complete": True,
                    "distributed_metrics": bool(distributed_section),
                    "final_throughput": perf_summary.get("avg_throughput_recent", 0),
                    "gradient_sync_efficiency": distributed_section.get("gradient_sync_time", 0)
                })
                return True
                
            finally:
                Path(model_path).unlink(missing_ok=True)
                
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Multi-GPU End-to-End Integration", False, duration, {"error": str(e)})
            return False
            
    async def run_all_multigpu_tests(self) -> Dict[str, Any]:
        """Run all multi-GPU Phase 3 integration tests."""
        
        print("=" * 90)
        print(f"🚀 PHASE 3 VERL INTEGRATION - MULTI-GPU TEST SUITE ({self.gpu_count} GPUs Available)")
        print("=" * 90)
        
        # Display system info
        if torch.cuda.is_available():
            print(f"🔥 CUDA Version: {torch.version.cuda}")
            for i in range(self.gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if self.ray_initialized:
            cluster_resources = ray.cluster_resources()
            print(f"⚡ Ray Cluster: {cluster_resources.get('GPU', 0)} GPUs, {cluster_resources.get('CPU', 0)} CPUs")
            
        print()
        
        test_functions = [
            ("Multi-GPU VERL Integration", self.test_multigpu_verl_integration),
            ("Parallel Evaluation Framework", self.test_parallel_evaluation_framework),
            ("Distributed Deployment Infrastructure", self.test_distributed_deployment_infrastructure),
            ("Multi-GPU Monitoring & Observability", self.test_multigpu_monitoring_observability),
            ("Multi-GPU End-to-End Integration", self.test_multigpu_end_to_end_integration)
        ]
        
        total_tests = len(test_functions)
        passed_tests = 0
        
        for test_name, test_func in test_functions:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print('='*70)
            
            success = await test_func()
            if success:
                passed_tests += 1
                
            # Clear GPU cache between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        total_duration = time.time() - self.start_time
        
        # Generate comprehensive multi-GPU report
        print("\n" + "=" * 90)
        print("📊 MULTI-GPU PHASE 3 INTEGRATION TEST RESULTS")
        print("=" * 90)
        
        # Test results summary
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["success"] else "❌"
            print(f"{status_icon} {test_name:<50} {result['duration']:>8.2f}s")
            
            # Show GPU utilization info
            gpu_info = result.get("gpu_info", {})
            if gpu_info.get("memory_info"):
                max_util = max((gpu["utilization_pct"] for gpu in gpu_info["memory_info"].values()), default=0)
                total_memory = sum((gpu["memory_used_gb"] for gpu in gpu_info["memory_info"].values()), default=0)
                print(f"    GPU: {max_util:.1f}% peak utilization, {total_memory:.1f} GB total memory used")
                
            if not result["success"] and "error" in result["details"]:
                print(f"    Error: {result['details']['error']}")
                
        print("-" * 90)
        print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"🔥 GPUs Available: {self.gpu_count}")
        
        if passed_tests == total_tests:
            print("🎉 ALL MULTI-GPU TESTS PASSED! Phase 3 is ready for production.")
            print(f"\n✨ Multi-GPU Phase 3 Implementation Status: COMPLETE ({self.gpu_count}x GPU) ✨")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
            
        print("=" * 90)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "total_duration": total_duration,
            "gpu_count": self.gpu_count,
            "test_results": self.test_results,
            "overall_success": passed_tests == total_tests
        }
        
    def save_multigpu_test_report(self, results: Dict[str, Any]) -> None:
        """Save detailed multi-GPU test report."""
        
        report_path = Path(f"phase3_multigpu_{self.gpu_count}gpu_test_report.json")
        
        # Gather additional system information
        system_info = {
            "gpu_count": self.gpu_count,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "ray_initialized": self.ray_initialized,
            "python_version": sys.version,
            "platform": sys.platform
        }
        
        if torch.cuda.is_available():
            gpu_details = {}
            for i in range(self.gpu_count):
                gpu_details[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    "compute_capability": torch.cuda.get_device_capability(i)
                }
            system_info["gpu_details"] = gpu_details
            
        if self.ray_initialized:
            try:
                cluster_resources = ray.cluster_resources()
                system_info["ray_cluster"] = {
                    "total_cpus": cluster_resources.get("CPU", 0),
                    "total_gpus": cluster_resources.get("GPU", 0),
                    "total_memory": cluster_resources.get("memory", 0)
                }
            except:
                pass
                
        detailed_report = {
            "test_run_timestamp": time.time(),
            "phase": "Phase 3 - Multi-GPU VERL Integration",
            "system_info": system_info,
            "configuration": {
                "verl_config": self.verl_config_data,
                "evaluation_config": self.eval_config_data,
                "deployment_config": self.deployment_config_data,
                "monitoring_config": self.monitoring_config_data
            },
            "summary": {
                "total_tests": results["total_tests"],
                "passed_tests": results["passed_tests"],
                "success_rate": results["success_rate"],
                "total_duration": results["total_duration"],
                "overall_success": results["overall_success"],
                "gpu_count": results["gpu_count"]
            },
            "test_results": results["test_results"]
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
            
        print(f"\n📋 Detailed multi-GPU test report saved to: {report_path}")


async def main():
    """Main multi-GPU test runner."""
    
    try:
        # Verify multi-GPU setup
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Multi-GPU testing requires CUDA.")
            sys.exit(1)
            
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            print(f"⚠️  Warning: Only {gpu_count} GPU detected. Multi-GPU features may not be fully tested.")
            
        # Verify Ray cluster
        if not ray.is_initialized():
            print("⚠️  Warning: Ray cluster not initialized. Run 'python configure_multigpu_tests.py' first.")
            
        tester = MultiGPUPhase3Tester()
        results = await tester.run_all_multigpu_tests()
        tester.save_multigpu_test_report(results)
        
        # Exit with appropriate code
        exit_code = 0 if results["overall_success"] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Multi-GPU tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical error in multi-GPU test runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set optimizations for multi-GPU testing
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable optimized cuDNN
    
    # Run the multi-GPU integration tests
    asyncio.run(main())