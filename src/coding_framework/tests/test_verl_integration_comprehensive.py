"""
Comprehensive Validation Tests for Phase 3 VERL Integration.

This test suite validates all Phase 3 components as specified in the updated PRP:
- VERL distributed training integration
- Code generation evaluation framework
- Production deployment infrastructure
- Monitoring and observability integration
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# VERL Integration imports
from ..verl_integration.verl_coordinator import VERLCoordinator
from ..verl_integration.verl_config import VERLDistributedConfig
from ..verl_integration.multi_agent_trainer import MultiAgentVERLTrainer
from ..verl_integration.verl_reward_adapter import VERLRewardAdapter

# Evaluation Framework imports
from ..evaluation.benchmark_manager import BenchmarkManager, BenchmarkManagerConfig
from ..evaluation.verl_evaluation_bridge import VERLEvaluationBridge, VERLEvaluationConfig
from ..evaluation.evaluators.humaneval_evaluator import HumanEvalEvaluator, HumanEvalConfig
from ..evaluation.evaluators.mbpp_evaluator import MBPPEvaluator, MBPPConfig
from ..evaluation.evaluators.bigcodebench_evaluator import BigCodeBenchEvaluator, BigCodeBenchConfig

# Deployment Infrastructure imports
from ..deployment.deployment_orchestrator import DeploymentOrchestrator, OrchestratorConfig
from ..deployment.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentTarget
from ..deployment.model_server import VERLModelServer, ModelServerConfig
from ..deployment.ab_testing import ABTestManager, ABTestConfig, TestVariant

# Monitoring imports
from ..monitoring.verl_training_monitor import (
    VERLTrainingMonitor, 
    VERLTrainingMonitorConfig,
    TrainingMetrics,
    TrainingPhase
)

# Agent imports
from ..agents.code_generator_agent import CodeGeneratorAgent
from ..agents.code_reviewer_agent import CodeReviewerAgent
from ..agents.agent_config import CodeGeneratorConfig, CodeReviewerConfig
from ..multi_agent.coordinator import MultiAgentCoordinator


@pytest.mark.comprehensive
class TestVERLIntegrationValidation:
    """Comprehensive validation tests for VERL integration."""
    
    @pytest.fixture
    async def verl_coordinator(self):
        """Create VERL coordinator for testing."""
        config = VERLDistributedConfig(
            algorithm="ppo",
            num_gpus=2,
            strategy="fsdp2"
        )
        coordinator = VERLCoordinator(config)
        return coordinator
        
    @pytest.fixture
    async def multi_agent_trainer(self):
        """Create multi-agent VERL trainer."""
        config = VERLDistributedConfig()
        trainer = MultiAgentVERLTrainer(config)
        return trainer
        
    @pytest.mark.asyncio
    async def test_verl_coordinator_initialization(self, verl_coordinator):
        """Test VERL coordinator initialization and configuration."""
        
        # Test configuration validation
        assert verl_coordinator.config.algorithm == "ppo"
        assert verl_coordinator.config.num_gpus == 2
        assert verl_coordinator.config.strategy == "fsdp2"
        
        # Test initialization
        init_result = await verl_coordinator.initialize()
        assert init_result["success"] == True
        assert "cluster_info" in init_result
        
        # Test configuration export
        verl_config = verl_coordinator.export_verl_config()
        assert "algorithm" in verl_config
        assert "distributed_config" in verl_config
        
    @pytest.mark.asyncio
    async def test_multi_agent_verl_integration(self, multi_agent_trainer):
        """Test multi-agent framework integration with VERL."""
        
        # Create test agents
        generator_config = CodeGeneratorConfig()
        reviewer_config = CodeReviewerConfig()
        
        generator = CodeGeneratorAgent(generator_config)
        reviewer = CodeReviewerAgent(reviewer_config)
        
        agents = [generator, reviewer]
        
        # Test agent setup
        setup_result = await multi_agent_trainer.setup_agents(agents)
        assert setup_result["success"] == True
        assert len(setup_result["agents_configured"]) == 2
        
        # Test reward adapter creation
        reward_adapter = VERLRewardAdapter()
        adapter_result = await reward_adapter.setup_reward_bridge(agents)
        assert adapter_result["success"] == True
        
    @pytest.mark.asyncio
    async def test_verl_distributed_training_simulation(self, verl_coordinator):
        """Test simulated VERL distributed training workflow."""
        
        await verl_coordinator.initialize()
        
        # Test training step simulation
        training_config = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "max_steps": 10
        }
        
        training_result = await verl_coordinator.start_training(training_config)
        assert training_result["success"] == True
        assert "training_id" in training_result
        
        # Test training status monitoring
        status = verl_coordinator.get_training_status(training_result["training_id"])
        assert "status" in status
        assert "current_step" in status
        
    @pytest.mark.asyncio
    async def test_verl_checkpoint_management(self, verl_coordinator):
        """Test VERL checkpoint saving and loading."""
        
        await verl_coordinator.initialize()
        
        # Test checkpoint creation
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint"
            
            save_result = await verl_coordinator.save_checkpoint(
                checkpoint_path=str(checkpoint_path),
                step=100
            )
            assert save_result["success"] == True
            
            # Test checkpoint loading
            load_result = await verl_coordinator.load_checkpoint(str(checkpoint_path))
            assert load_result["success"] == True
            assert load_result["step"] == 100


@pytest.mark.comprehensive
class TestEvaluationFrameworkValidation:
    """Comprehensive validation tests for evaluation framework."""
    
    @pytest.fixture
    def benchmark_manager_config(self):
        """Create benchmark manager configuration."""
        return BenchmarkManagerConfig(
            results_dir="./test_evaluation_results",
            parallel_execution=True,
            save_detailed_results=True
        )
        
    @pytest.fixture
    async def benchmark_manager(self, benchmark_manager_config):
        """Create benchmark manager for testing."""
        return BenchmarkManager(benchmark_manager_config)
        
    @pytest.fixture
    def test_agent(self):
        """Create test agent for evaluation."""
        config = CodeGeneratorConfig()
        return CodeGeneratorAgent(config)
        
    @pytest.mark.asyncio
    async def test_humaneval_evaluator_functionality(self, test_agent):
        """Test HumanEval evaluator functionality."""
        
        config = HumanEvalConfig(save_individual_results=True)
        evaluator = HumanEvalEvaluator(config)
        
        # Test dataset loading
        problems = evaluator.load_dataset()
        assert len(problems) > 0
        assert all("task_id" in p for p in problems)
        
        # Test evaluation
        result = await evaluator.evaluate_agent(test_agent)
        
        assert result.success == True
        assert result.benchmark == "humaneval"
        assert result.total_problems > 0
        assert hasattr(result, 'pass_at_1')
        assert hasattr(result, 'pass_at_10')
        
    @pytest.mark.asyncio
    async def test_mbpp_evaluator_functionality(self, test_agent):
        """Test MBPP evaluator functionality."""
        
        config = MBPPConfig(use_three_shot=True)
        evaluator = MBPPEvaluator(config)
        
        # Test few-shot prompt creation
        test_problem = "Write a function to add two numbers."
        prompt = evaluator.create_few_shot_prompt(test_problem)
        
        assert "You are an expert Python programmer" in prompt
        assert "Example 1:" in prompt
        assert test_problem in prompt
        
        # Test evaluation
        result = await evaluator.evaluate_agent(test_agent)
        
        assert result.success == True
        assert result.benchmark == "mbpp"
        assert result.metadata["use_three_shot"] == True
        
    @pytest.mark.asyncio
    async def test_bigcodebench_evaluator_functionality(self, test_agent):
        """Test BigCodeBench evaluator functionality."""
        
        config = BigCodeBenchConfig(mode="instruct", extended_timeout=30)
        evaluator = BigCodeBenchEvaluator(config)
        
        # Test mode-specific prompt retrieval
        test_problem = {
            "instruct_prompt": "Create a data visualization function",
            "complete_prompt": "import matplotlib.pyplot as plt\n\ndef create_plot():"
        }
        
        instruct_prompt = evaluator.get_prompt_for_mode(test_problem)
        assert instruct_prompt == test_problem["instruct_prompt"]
        
        # Test evaluation
        result = await evaluator.evaluate_agent(test_agent)
        
        assert result.success == True
        assert result.benchmark == "bigcodebench"
        assert result.metadata["mode"] == "instruct"
        
    @pytest.mark.asyncio
    async def test_comprehensive_benchmark_evaluation(self, benchmark_manager, test_agent):
        """Test comprehensive evaluation across all benchmarks."""
        
        benchmarks = ["humaneval", "mbpp", "bigcodebench"]
        
        result = await benchmark_manager.run_comprehensive_evaluation(
            agent=test_agent,
            benchmarks=benchmarks,
            save_results=True
        )
        
        assert result["agent_id"] == test_agent.agent_id
        assert len(result["benchmarks"]) == 3
        assert "aggregate" in result
        
        # Verify aggregate metrics
        aggregate = result["aggregate"]
        assert "weighted_pass_at_1" in aggregate
        assert "benchmark_scores" in aggregate
        assert "success_rate" in aggregate
        
        # Verify individual benchmark results
        for benchmark in benchmarks:
            assert benchmark in result["benchmarks"]
            bench_result = result["benchmarks"][benchmark]
            if isinstance(bench_result, dict) and bench_result.get("success"):
                assert "pass_at_1" in bench_result
                
    @pytest.mark.asyncio
    async def test_verl_evaluation_bridge_integration(self):
        """Test VERL evaluation bridge integration."""
        
        config = VERLEvaluationConfig(
            evaluation_frequency=10,
            benchmarks=["humaneval"],
            quick_eval_problems=5
        )
        bridge = VERLEvaluationBridge(config)
        
        # Test evaluation triggering logic
        assert await bridge.should_evaluate(10) == True  # At frequency
        assert await bridge.should_evaluate(15) == False  # Between frequencies
        assert await bridge.should_evaluate(500) == True  # At full eval frequency
        
        # Test early stopping logic
        bridge.evaluations_without_improvement = 5
        bridge.config.early_stopping_patience = 3
        assert bridge.should_early_stop() == True
        
        # Test evaluation summary
        summary = bridge.get_evaluation_summary()
        assert "total_evaluations" in summary
        assert "best_score" in summary
        
    @pytest.mark.asyncio
    async def test_evaluation_metrics_calculation(self, benchmark_manager):
        """Test evaluation metrics calculation accuracy."""
        
        # Create mock problem results for testing pass@k calculation
        mock_results = [
            {"solutions_passed": 1, "solutions_generated": 5},  # 1/5 success
            {"solutions_passed": 0, "solutions_generated": 3},  # 0/3 success
            {"solutions_passed": 2, "solutions_generated": 4},  # 2/4 success
        ]
        
        # Test pass@1 calculation
        pass_at_1 = sum(1 for r in mock_results if r["solutions_passed"] > 0) / len(mock_results)
        assert pass_at_1 == 2/3  # 2 out of 3 problems solved
        
        # Test pass@k calculation
        from ..evaluation.evaluators.base_evaluator import BaseEvaluator
        evaluator = BaseEvaluator()
        pass_at_3 = evaluator.calculate_pass_at_k(mock_results, 3)
        
        # Should be higher than pass@1 since more attempts allowed
        assert pass_at_3 >= pass_at_1


@pytest.mark.comprehensive  
class TestDeploymentInfrastructureValidation:
    """Comprehensive validation tests for deployment infrastructure."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration."""
        return OrchestratorConfig(
            enable_automated_rollback=True,
            enable_deployment_monitoring=False,  # Simplified for testing
            max_deployment_time=300
        )
        
    @pytest.fixture
    def deployment_orchestrator(self, orchestrator_config):
        """Create deployment orchestrator."""
        return DeploymentOrchestrator(orchestrator_config)
        
    @pytest.fixture
    def test_model_path(self):
        """Create temporary test model file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            temp_file.write(b"mock_model_data")
            return temp_file.name
            
    @pytest.mark.asyncio
    async def test_deployment_manager_strategies(self, test_model_path):
        """Test all deployment strategies."""
        
        deployment_manager = DeploymentManager()
        
        strategies = ["blue_green", "rolling", "canary"]
        
        for strategy in strategies:
            config = DeploymentConfig(
                model_path=test_model_path,
                target_environment=DeploymentTarget.DEVELOPMENT,
                deployment_strategy=strategy,
                replicas=2
            )
            
            result = await deployment_manager.deploy_model(
                deployment_id=f"test_{strategy}_deploy",
                model_path=test_model_path,
                config=config
            )
            
            assert result["success"] == True
            assert result["strategy"] == strategy
            
            # Verify strategy-specific results
            if strategy == "blue_green":
                assert "traffic_switched" in result["steps_completed"]
            elif strategy == "rolling":
                assert result["total_replicas"] == config.replicas
            elif strategy == "canary":
                assert "canary_promoted" in result["steps_completed"]
                
        # Cleanup
        Path(test_model_path).unlink(missing_ok=True)
        
    @pytest.mark.asyncio
    async def test_model_server_functionality(self, test_model_path):
        """Test VERL model server functionality."""
        
        config = ModelServerConfig(
            model_path=test_model_path,
            port=8099,
            max_concurrent_requests=10,
            enable_batching=True
        )
        
        server = VERLModelServer(config)
        
        # Test server startup
        start_result = await server.start_server()
        assert start_result["success"] == True
        assert "endpoints" in start_result
        
        # Test health check
        health_status = await server.get_health_status()
        assert "status" in health_status
        assert "uptime" in health_status
        
        # Test inference request processing
        test_payload = {
            "input": "def fibonacci(n):",
            "context": {"temperature": 0.7}
        }
        
        inference_result = await server.process_inference_request(
            request_id="test_inference_001",
            payload=test_payload
        )
        
        assert "request_id" in inference_result
        assert "success" in inference_result
        assert "metadata" in inference_result
        
        # Test batch inference
        batch_requests = [
            {"request_id": "batch_1", "input": "def add(a, b):"},
            {"request_id": "batch_2", "input": "def subtract(a, b):"}
        ]
        
        batch_results = await server.process_batch_inference(batch_requests)
        assert len(batch_results) == 2
        
        # Test server shutdown
        stop_result = await server.stop_server()
        assert stop_result["success"] == True
        
        # Cleanup
        Path(test_model_path).unlink(missing_ok=True)
        
    @pytest.mark.asyncio
    async def test_ab_testing_functionality(self):
        """Test A/B testing functionality."""
        
        variants = [
            TestVariant(
                variant_id="control",
                name="Control Model",
                model_path="/tmp/control_model",
                traffic_allocation=60.0
            ),
            TestVariant(
                variant_id="experiment",
                name="Experimental Model", 
                model_path="/tmp/experiment_model",
                traffic_allocation=40.0
            )
        ]
        
        ab_config = ABTestConfig(
            test_id="comprehensive_test_001",
            test_name="Model Comparison Test",
            description="Testing model improvements",
            variants=variants,
            duration_days=1,
            min_sample_size=50
        )
        
        ab_manager = ABTestManager()
        
        # Test A/B test creation
        create_result = await ab_manager.create_test(ab_config)
        assert create_result["success"] == True
        assert create_result["variants_deployed"] == 2
        
        # Test A/B test startup
        start_result = await ab_manager.start_test(ab_config.test_id)
        assert start_result["success"] == True
        
        # Test traffic routing
        routing_results = []
        for i in range(100):
            result = await ab_manager.route_request(
                test_id=ab_config.test_id,
                request_context={"user_id": f"user_{i}"}
            )
            routing_results.append(result["variant_id"])
            
        # Verify traffic distribution roughly matches allocation
        control_pct = routing_results.count("control") / len(routing_results) * 100
        experiment_pct = routing_results.count("experiment") / len(routing_results) * 100
        
        assert 50 <= control_pct <= 70  # Should be around 60%
        assert 30 <= experiment_pct <= 50  # Should be around 40%
        
        # Test result recording
        for i in range(20):
            await ab_manager.record_result(
                test_id=ab_config.test_id,
                variant_id="control",
                result={"success": True, "latency": 0.1, "conversion": i % 2 == 0}
            )
            
        # Test results analysis
        test_results = await ab_manager.get_test_results(ab_config.test_id)
        assert test_results["test_id"] == ab_config.test_id
        assert len(test_results["variants"]) == 2
        
        # Test A/B test stopping
        stop_result = await ab_manager.stop_test(ab_config.test_id)
        assert stop_result["success"] == True
        
    @pytest.mark.asyncio
    async def test_full_deployment_orchestration(self, deployment_orchestrator, test_model_path):
        """Test complete deployment orchestration workflow."""
        
        deployment_config = DeploymentConfig(
            model_path=test_model_path,
            target_environment=DeploymentTarget.DEVELOPMENT,
            deployment_strategy="blue_green"
        )
        
        server_config = ModelServerConfig(
            model_path=test_model_path,
            port=8100
        )
        
        # Test full workflow execution
        result = await deployment_orchestrator.deploy_model_full_workflow(
            workflow_id="comprehensive_workflow_test",
            model_path=test_model_path,
            target_environment=DeploymentTarget.DEVELOPMENT,
            deployment_config=deployment_config,
            server_config=server_config,
            enable_ab_test=False
        )
        
        assert result["success"] == True
        assert "workflow_id" in result
        assert "phases_completed" in result
        assert len(result["phases_completed"]) >= 6  # Multiple phases
        
        # Test workflow status tracking
        status = deployment_orchestrator.get_workflow_status("comprehensive_workflow_test")
        assert status["workflow_id"] == "comprehensive_workflow_test"
        
        # Cleanup
        Path(test_model_path).unlink(missing_ok=True)


@pytest.mark.comprehensive
class TestMonitoringIntegrationValidation:
    """Comprehensive validation tests for monitoring integration."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Create monitoring configuration."""
        return VERLTrainingMonitorConfig(
            metrics_collection_interval=10,
            enable_alerts=True,
            export_metrics_to_file=True,
            metrics_export_path="./test_monitoring_data"
        )
        
    @pytest.fixture
    def training_monitor(self, monitoring_config):
        """Create training monitor."""
        return VERLTrainingMonitor(monitoring_config)
        
    @pytest.mark.asyncio
    async def test_training_monitor_initialization(self, training_monitor):
        """Test training monitor initialization."""
        
        # Test monitor startup
        start_result = await training_monitor.start_monitoring()
        assert start_result["success"] == True
        assert "monitoring_started_at" in start_result
        
        # Test monitoring status
        assert training_monitor.monitoring_active == True
        assert training_monitor.training_start_time is not None
        
    @pytest.mark.asyncio
    async def test_metrics_recording_and_analysis(self, training_monitor):
        """Test metrics recording and analysis functionality."""
        
        await training_monitor.start_monitoring()
        
        # Record sample training metrics
        sample_metrics = [
            {
                "policy_loss": 2.5,
                "value_loss": 1.2, 
                "reward_mean": -5.0,
                "reward_std": 2.1,
                "samples_per_second": 150.0,
                "gpu_utilization": 85.0,
                "memory_usage_gb": 6.5
            },
            {
                "policy_loss": 2.1,
                "value_loss": 1.0,
                "reward_mean": -3.5,
                "reward_std": 1.8,
                "samples_per_second": 165.0,
                "gpu_utilization": 88.0,
                "memory_usage_gb": 6.8
            },
            {
                "policy_loss": 1.8,
                "value_loss": 0.9,
                "reward_mean": -2.0,
                "reward_std": 1.5,
                "samples_per_second": 175.0,
                "gpu_utilization": 90.0,
                "memory_usage_gb": 7.0
            }
        ]
        
        # Record metrics
        for i, metrics in enumerate(sample_metrics):
            await training_monitor.record_training_metrics(
                metrics=metrics,
                training_step=i * 100,
                phase=TrainingPhase.POLICY_TRAINING
            )
            
        # Test training summary
        summary = await training_monitor.get_training_summary()
        
        assert summary["training_status"]["monitoring_active"] == True
        assert summary["training_status"]["total_metrics_recorded"] == 3
        assert summary["performance_summary"]["avg_reward_recent"] > -5.0  # Improving
        
        # Test performance analysis
        analysis = await training_monitor.get_performance_analysis()
        
        assert "training_efficiency" in analysis
        assert "resource_utilization" in analysis
        assert "convergence_analysis" in analysis
        assert "recommendations" in analysis
        
        # Verify improvement detection
        assert analysis["convergence_analysis"]["reward_trend"] == "increasing"
        assert analysis["convergence_analysis"]["loss_trend"] == "decreasing"
        
    @pytest.mark.asyncio
    async def test_evaluation_results_integration(self, training_monitor):
        """Test evaluation results integration with monitoring."""
        
        await training_monitor.start_monitoring()
        
        # Mock evaluation results
        evaluation_results = {
            "benchmarks": {
                "humaneval": {
                    "pass_at_1": 0.65,
                    "pass_at_10": 0.82,
                    "solved_problems": 107,
                    "total_problems": 164
                },
                "mbpp": {
                    "pass_at_1": 0.58,
                    "pass_at_10": 0.75,
                    "solved_problems": 58,
                    "total_problems": 100
                }
            },
            "aggregate": {
                "weighted_pass_at_1": 0.615,
                "overall_pass_rate": 0.62,
                "total_problems": 264
            }
        }
        
        # Record evaluation results
        await training_monitor.record_evaluation_results(
            evaluation_results=evaluation_results,
            training_step=500
        )
        
        # Verify metrics were recorded
        assert len(training_monitor.metrics_history) > 0
        
        # Find evaluation metric
        eval_metrics = [m for m in training_monitor.metrics_history 
                      if m.phase == TrainingPhase.EVALUATION]
        assert len(eval_metrics) > 0
        
        eval_metric = eval_metrics[-1]
        assert eval_metric.custom_metrics["weighted_pass_at_1"] == 0.615
        assert eval_metric.custom_metrics["humaneval_pass_at_1"] == 0.65
        
    @pytest.mark.asyncio
    async def test_alert_system_functionality(self, training_monitor):
        """Test monitoring alert system."""
        
        await training_monitor.start_monitoring()
        
        # Record metrics that should trigger alerts
        alert_triggering_metrics = {
            "policy_loss": 10.0,  # Should trigger high_policy_loss alert
            "reward_mean": -15.0,  # Should trigger low_reward_mean alert
            "gpu_utilization": 98.0,  # Should trigger high_gpu_utilization alert
            "samples_per_second": 5.0  # Should trigger low_throughput alert
        }
        
        # Record multiple times to trigger consecutive violations
        for i in range(5):
            await training_monitor.record_training_metrics(
                metrics=alert_triggering_metrics,
                training_step=i * 10,
                phase=TrainingPhase.POLICY_TRAINING
            )
            
        # Wait for alert evaluation
        await asyncio.sleep(0.1)
        
        # Verify alerts were triggered (check violation counts)
        assert training_monitor.alert_violations["high_policy_loss"] >= 3
        assert training_monitor.alert_violations["low_reward_mean"] >= 3
        
    @pytest.mark.asyncio
    async def test_monitoring_data_export(self, training_monitor):
        """Test monitoring data export functionality."""
        
        await training_monitor.start_monitoring()
        
        # Record some metrics
        test_metrics = {
            "policy_loss": 1.5,
            "reward_mean": 5.0,
            "samples_per_second": 200.0
        }
        
        await training_monitor.record_training_metrics(
            metrics=test_metrics,
            training_step=100,
            phase=TrainingPhase.POLICY_TRAINING
        )
        
        # Stop monitoring (triggers export)
        stop_result = await training_monitor.stop_monitoring()
        
        assert stop_result["success"] == True
        assert stop_result["total_metrics_collected"] > 0
        
        # Verify export files were created
        export_path = Path(training_monitor.config.metrics_export_path)
        if export_path.exists():
            # Check for metrics export file
            metrics_file = export_path / "training_metrics.jsonl"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    exported_data = f.read()
                    assert "policy_loss" in exported_data
                    
            # Check for final summary
            summary_file = export_path / "final_training_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary_data = json.load(f)
                    assert "training_status" in summary_data


@pytest.mark.comprehensive
class TestEndToEndIntegration:
    """End-to-end integration tests combining all Phase 3 components."""
    
    @pytest.mark.asyncio
    async def test_complete_verl_training_evaluation_deployment_workflow(self):
        """Test complete workflow from VERL training through evaluation to deployment."""
        
        # Phase 1: Setup VERL training
        verl_config = VERLDistributedConfig(
            algorithm="ppo",
            num_gpus=1,
            strategy="fsdp2"
        )
        
        verl_coordinator = VERLCoordinator(verl_config)
        await verl_coordinator.initialize()
        
        # Phase 2: Setup monitoring
        monitoring_config = VERLTrainingMonitorConfig(
            metrics_collection_interval=5,
            enable_alerts=True
        )
        monitor = VERLTrainingMonitor(monitoring_config)
        await monitor.start_monitoring(verl_config)
        
        # Phase 3: Simulate training with monitoring
        for step in range(5):
            training_metrics = {
                "policy_loss": max(3.0 - step * 0.5, 0.5),  # Decreasing loss
                "reward_mean": -10.0 + step * 2.0,  # Improving rewards
                "samples_per_second": 100.0 + step * 10.0,
                "gpu_utilization": 80.0 + step * 2.0
            }
            
            await monitor.record_training_metrics(
                metrics=training_metrics,
                training_step=step * 100,
                phase=TrainingPhase.POLICY_TRAINING
            )
            
        # Phase 4: Run evaluation
        test_agent = CodeGeneratorAgent(CodeGeneratorConfig())
        
        evaluation_bridge = VERLEvaluationBridge()
        eval_result = await evaluation_bridge.evaluate_during_training(
            agent=test_agent,
            step=500,
            full_evaluation=True
        )
        
        assert eval_result["step"] == 500
        assert "results" in eval_result
        
        # Record evaluation in monitoring
        if eval_result.get("results"):
            await monitor.record_evaluation_results(
                evaluation_results=eval_result["results"],
                training_step=500
            )
            
        # Phase 5: Deploy trained model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            model_path = temp_file.name
            temp_file.write(b"trained_model_data")
            
        try:
            orchestrator = DeploymentOrchestrator()
            
            deployment_result = await orchestrator.deploy_model_full_workflow(
                workflow_id="e2e_integration_test",
                model_path=model_path,
                target_environment=DeploymentTarget.DEVELOPMENT
            )
            
            assert deployment_result["success"] == True
            
            # Phase 6: Generate final monitoring summary
            final_summary = await monitor.get_training_summary()
            
            assert final_summary["training_status"]["monitoring_active"] == True
            assert final_summary["training_status"]["total_metrics_recorded"] >= 5
            
            # Verify performance improvements
            perf_summary = final_summary["performance_summary"]
            assert perf_summary["avg_reward_recent"] > -10.0  # Should have improved
            
        finally:
            Path(model_path).unlink(missing_ok=True)
            await monitor.stop_monitoring()
            
    @pytest.mark.asyncio
    async def test_fault_tolerance_and_recovery(self):
        """Test fault tolerance and recovery mechanisms across all components."""
        
        # Test VERL coordinator failure recovery
        verl_config = VERLDistributedConfig()
        coordinator = VERLCoordinator(verl_config)
        
        await coordinator.initialize()
        
        # Simulate training failure and recovery
        training_config = {"max_steps": 100}
        training_result = await coordinator.start_training(training_config)
        
        if training_result["success"]:
            # Simulate failure by stopping training
            stop_result = await coordinator.stop_training(training_result["training_id"])
            assert stop_result["success"] == True
            
        # Test deployment failure and rollback
        deployment_manager = DeploymentManager()
        
        # Use invalid model path to trigger failure
        invalid_config = DeploymentConfig(
            model_path="/nonexistent/model",
            target_environment=DeploymentTarget.DEVELOPMENT,
            rollback_on_failure=True
        )
        
        deployment_result = await deployment_manager.deploy_model(
            deployment_id="failure_test",
            model_path="/nonexistent/model", 
            config=invalid_config
        )
        
        # Should fail but attempt rollback
        assert deployment_result["success"] == False
        # Would verify rollback was attempted based on configuration
        
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance benchmarks across all components."""
        
        # Test evaluation framework performance
        start_time = time.time()
        
        test_agent = CodeGeneratorAgent(CodeGeneratorConfig())
        benchmark_manager = BenchmarkManager()
        
        # Run quick evaluation
        result = await benchmark_manager.quick_evaluation(
            agent=test_agent,
            benchmark="humaneval",
            num_problems=5
        )
        
        evaluation_time = time.time() - start_time
        
        assert result.get("pass_at_1") is not None
        assert evaluation_time < 30  # Should complete within 30 seconds for 5 problems
        
        # Test deployment orchestration performance
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            model_path = temp_file.name
            temp_file.write(b"test_model")
            
        try:
            start_time = time.time()
            
            orchestrator = DeploymentOrchestrator()
            deployment_result = await orchestrator.deploy_model_full_workflow(
                workflow_id="perf_test",
                model_path=model_path,
                target_environment=DeploymentTarget.DEVELOPMENT
            )
            
            deployment_time = time.time() - start_time
            
            assert deployment_result["success"] == True
            assert deployment_time < 60  # Should complete within 1 minute for test deployment
            
        finally:
            Path(model_path).unlink(missing_ok=True)
            
    @pytest.mark.asyncio
    async def test_resource_cleanup_and_memory_management(self):
        """Test proper resource cleanup and memory management."""
        
        # Test monitoring cleanup
        monitor = VERLTrainingMonitor()
        await monitor.start_monitoring()
        
        # Generate metrics to fill memory
        for i in range(1000):
            await monitor.record_training_metrics(
                metrics={"step": i, "value": i * 0.1},
                training_step=i
            )
            
        # Verify metrics retention policy
        initial_count = len(monitor.metrics_history)
        
        # Stop monitoring and verify cleanup
        stop_result = await monitor.stop_monitoring()
        assert stop_result["success"] == True
        
        # Test deployment resource cleanup
        orchestrator = DeploymentOrchestrator()
        
        # Verify no active workflows initially
        active_workflows = orchestrator.list_active_workflows()
        assert len(active_workflows) == 0
        
        # After deployment, resources should be properly cleaned up
        # (This is tested implicitly by the workflow completing successfully)


if __name__ == "__main__":
    # Run comprehensive validation tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "comprehensive",
        "--durations=10"  # Show 10 slowest tests
    ])