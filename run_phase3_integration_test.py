#!/usr/bin/env python3
"""
Phase 3 End-to-End Integration Test Runner

This script runs comprehensive integration tests for all Phase 3 components:
1. VERL distributed training integration
2. Code generation evaluation framework  
3. Production deployment infrastructure
4. Enhanced monitoring and observability

Run this script to validate the complete Phase 3 implementation.
"""

import asyncio
import sys
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List

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


class Phase3IntegrationTester:
    """Comprehensive integration tester for Phase 3 implementation."""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        
    def log_test_progress(self, test_name: str, status: str, details: str = ""):
        """Log test progress."""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.1f}s] {test_name}: {status}")
        if details:
            print(f"    {details}")
            
    def record_test_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any] = None):
        """Record test result."""
        self.test_results[test_name] = {
            "success": success,
            "duration": duration,
            "details": details or {}
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        self.log_test_progress(test_name, f"{status} ({duration:.2f}s)")
        
    async def test_verl_integration_components(self) -> bool:
        """Test VERL integration components."""
        
        test_start = time.time()
        self.log_test_progress("VERL Integration Test", "STARTING")
        
        try:
            # Test VERL Coordinator
            verl_config = VERLDistributedConfig(
                algorithm="ppo",
                num_gpus=1,  # Single GPU for testing
                strategy="fsdp2",
                ray_config={
                    "num_workers": 2,
                    "num_cpus_per_worker": 1
                }
            )
            
            coordinator = VERLCoordinator(verl_config)
            
            # Test initialization
            init_result = await coordinator.initialize()
            if not init_result.get("success", False):
                raise Exception(f"VERL initialization failed: {init_result}")
                
            # Test configuration export
            exported_config = coordinator.export_verl_config()
            if "algorithm" not in exported_config:
                raise Exception("VERL config export missing algorithm")
                
            # Test Multi-Agent VERL Trainer
            trainer = MultiAgentVERLTrainer(verl_config)
            
            # Create test agents
            generator_config = CodeGeneratorConfig()
            test_agent = CodeGeneratorAgent(generator_config)
            
            setup_result = await trainer.setup_agents([test_agent])
            if not setup_result.get("success", False):
                raise Exception(f"Multi-agent setup failed: {setup_result}")
                
            # Test simulated training step
            training_config = {
                "batch_size": 8,
                "learning_rate": 1e-4,
                "max_steps": 3
            }
            
            training_result = await coordinator.start_training(training_config)
            if not training_result.get("success", False):
                raise Exception(f"Training start failed: {training_result}")
                
            duration = time.time() - test_start
            self.record_test_result("VERL Integration", True, duration, {
                "coordinator_initialized": True,
                "multi_agent_setup": True,
                "training_started": True
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("VERL Integration", False, duration, {"error": str(e)})
            return False
            
    async def test_evaluation_framework(self) -> bool:
        """Test code generation evaluation framework."""
        
        test_start = time.time()
        self.log_test_progress("Evaluation Framework Test", "STARTING")
        
        try:
            # Setup benchmark manager
            config = BenchmarkManagerConfig(
                results_dir="./test_results",
                parallel_execution=False,  # Sequential for testing
                save_detailed_results=True
            )
            
            benchmark_manager = BenchmarkManager(config)
            
            # Create test agent
            agent_config = CodeGeneratorConfig()
            test_agent = CodeGeneratorAgent(agent_config)
            
            # Test quick evaluation on HumanEval
            quick_eval_result = await benchmark_manager.quick_evaluation(
                agent=test_agent,
                benchmark="humaneval",
                num_problems=3  # Small number for testing
            )
            
            if not hasattr(quick_eval_result, 'success') or not quick_eval_result.success:
                raise Exception(f"Quick evaluation failed: {quick_eval_result}")
                
            # Test comprehensive evaluation
            comprehensive_result = await benchmark_manager.run_comprehensive_evaluation(
                agent=test_agent,
                benchmarks=["humaneval"],  # Single benchmark for testing
                save_results=True
            )
            
            if "agent_id" not in comprehensive_result:
                raise Exception(f"Comprehensive evaluation failed: {comprehensive_result}")
                
            # Test VERL evaluation bridge
            bridge_config = VERLEvaluationConfig(
                evaluation_frequency=10,
                benchmarks=["humaneval"],
                quick_eval_problems=2
            )
            
            eval_bridge = VERLEvaluationBridge(bridge_config)
            
            # Test evaluation during training
            bridge_result = await eval_bridge.evaluate_during_training(
                agent=test_agent,
                step=100,
                full_evaluation=False
            )
            
            if bridge_result.get("step") != 100:
                raise Exception(f"Evaluation bridge failed: {bridge_result}")
                
            duration = time.time() - test_start
            self.record_test_result("Evaluation Framework", True, duration, {
                "quick_evaluation": True,
                "comprehensive_evaluation": True,
                "verl_bridge": True,
                "pass_at_1": quick_eval_result.pass_at_1
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Evaluation Framework", False, duration, {"error": str(e)})
            return False
            
    async def test_deployment_infrastructure(self) -> bool:
        """Test production deployment infrastructure."""
        
        test_start = time.time()
        self.log_test_progress("Deployment Infrastructure Test", "STARTING")
        
        try:
            # Create temporary model file
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
                model_path = temp_file.name
                temp_file.write(b"mock_trained_model_data")
                
            try:
                # Test deployment orchestrator
                orchestrator_config = OrchestratorConfig(
                    enable_automated_rollback=True,
                    enable_deployment_monitoring=False,  # Simplified for testing
                    max_deployment_time=180  # 3 minutes for testing
                )
                
                orchestrator = DeploymentOrchestrator(orchestrator_config)
                
                # Test full deployment workflow
                deployment_result = await orchestrator.deploy_model_full_workflow(
                    workflow_id="integration_test_deployment",
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT
                )
                
                if not deployment_result.get("success", False):
                    raise Exception(f"Deployment workflow failed: {deployment_result}")
                    
                # Test workflow status tracking
                status = orchestrator.get_workflow_status("integration_test_deployment")
                if "workflow_id" not in status:
                    raise Exception(f"Workflow status tracking failed: {status}")
                    
                # Test deployment health check
                health_result = await orchestrator.get_deployment_health("integration_test_deployment")
                if "workflow_id" not in health_result:
                    raise Exception(f"Health check failed: {health_result}")
                    
                duration = time.time() - test_start
                self.record_test_result("Deployment Infrastructure", True, duration, {
                    "workflow_completed": True,
                    "status_tracking": True,
                    "health_check": True,
                    "phases_completed": len(deployment_result.get("phases_completed", []))
                })
                return True
                
            finally:
                # Cleanup temporary file
                Path(model_path).unlink(missing_ok=True)
                
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Deployment Infrastructure", False, duration, {"error": str(e)})
            return False
            
    async def test_monitoring_observability(self) -> bool:
        """Test monitoring and observability integration."""
        
        test_start = time.time()
        self.log_test_progress("Monitoring & Observability Test", "STARTING")
        
        try:
            # Setup training monitor
            monitor_config = VERLTrainingMonitorConfig(
                metrics_collection_interval=5,
                enable_alerts=True,
                export_metrics_to_file=True,
                metrics_export_path="./test_monitoring_output"
            )
            
            monitor = VERLTrainingMonitor(monitor_config)
            
            # Test monitor startup
            start_result = await monitor.start_monitoring()
            if not start_result.get("success", False):
                raise Exception(f"Monitor startup failed: {start_result}")
                
            # Test metrics recording
            test_metrics = [
                {
                    "policy_loss": 3.2,
                    "value_loss": 1.5,
                    "reward_mean": -8.5,
                    "samples_per_second": 120.0,
                    "gpu_utilization": 75.0
                },
                {
                    "policy_loss": 2.8,
                    "value_loss": 1.3, 
                    "reward_mean": -6.2,
                    "samples_per_second": 135.0,
                    "gpu_utilization": 82.0
                },
                {
                    "policy_loss": 2.1,
                    "value_loss": 1.0,
                    "reward_mean": -4.1,
                    "samples_per_second": 150.0,
                    "gpu_utilization": 85.0
                }
            ]
            
            # Record training metrics
            for i, metrics in enumerate(test_metrics):
                await monitor.record_training_metrics(
                    metrics=metrics,
                    training_step=i * 100,
                    phase=TrainingPhase.POLICY_TRAINING
                )
                
            # Test evaluation results integration
            eval_results = {
                "benchmarks": {
                    "humaneval": {
                        "pass_at_1": 0.42,
                        "pass_at_10": 0.67,
                        "solved_problems": 69,
                        "total_problems": 164
                    }
                },
                "aggregate": {
                    "weighted_pass_at_1": 0.42,
                    "overall_pass_rate": 0.42
                }
            }
            
            await monitor.record_evaluation_results(
                evaluation_results=eval_results,
                training_step=300
            )
            
            # Test training summary
            summary = await monitor.get_training_summary()
            if not summary.get("training_status", {}).get("monitoring_active"):
                raise Exception("Training summary shows monitoring inactive")
                
            # Test performance analysis
            analysis = await monitor.get_performance_analysis()
            if "training_efficiency" not in analysis:
                raise Exception("Performance analysis missing key components")
                
            # Test monitor shutdown
            stop_result = await monitor.stop_monitoring()
            if not stop_result.get("success", False):
                raise Exception(f"Monitor shutdown failed: {stop_result}")
                
            duration = time.time() - test_start
            self.record_test_result("Monitoring & Observability", True, duration, {
                "monitor_startup": True,
                "metrics_recording": True,
                "evaluation_integration": True,
                "performance_analysis": True,
                "total_metrics": stop_result.get("total_metrics_collected", 0)
            })
            return True
            
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("Monitoring & Observability", False, duration, {"error": str(e)})
            return False
            
    async def test_end_to_end_integration(self) -> bool:
        """Test complete end-to-end integration workflow."""
        
        test_start = time.time()
        self.log_test_progress("End-to-End Integration Test", "STARTING")
        
        try:
            # Phase 1: Initialize all components
            verl_config = VERLDistributedConfig(algorithm="ppo", num_gpus=1)
            coordinator = VERLCoordinator(verl_config)
            await coordinator.initialize()
            
            monitor = VERLTrainingMonitor()
            await monitor.start_monitoring(verl_config)
            
            # Phase 2: Simulate training with monitoring
            for step in range(3):
                metrics = {
                    "policy_loss": 3.0 - step * 0.8,
                    "reward_mean": -10.0 + step * 3.0,
                    "samples_per_second": 100.0 + step * 20.0
                }
                
                await monitor.record_training_metrics(
                    metrics=metrics,
                    training_step=step * 50,
                    phase=TrainingPhase.POLICY_TRAINING
                )
                
            # Phase 3: Run evaluation
            test_agent = CodeGeneratorAgent(CodeGeneratorConfig())
            eval_bridge = VERLEvaluationBridge()
            
            eval_result = await eval_bridge.evaluate_during_training(
                agent=test_agent,
                step=150,
                full_evaluation=False
            )
            
            # Record evaluation in monitoring
            if eval_result.get("results"):
                await monitor.record_evaluation_results(
                    evaluation_results=eval_result["results"],
                    training_step=150
                )
                
            # Phase 4: Deploy model
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
                model_path = temp_file.name
                temp_file.write(b"e2e_trained_model")
                
            try:
                orchestrator = DeploymentOrchestrator()
                deploy_result = await orchestrator.deploy_model_full_workflow(
                    workflow_id="e2e_integration",
                    model_path=model_path,
                    target_environment=DeploymentTarget.DEVELOPMENT
                )
                
                if not deploy_result.get("success"):
                    raise Exception(f"E2E deployment failed: {deploy_result}")
                    
                # Phase 5: Final monitoring summary
                final_summary = await monitor.get_training_summary()
                
                await monitor.stop_monitoring()
                
                duration = time.time() - test_start
                self.record_test_result("End-to-End Integration", True, duration, {
                    "training_simulation": True,
                    "evaluation_integration": True, 
                    "deployment_success": True,
                    "monitoring_complete": True,
                    "total_metrics": final_summary.get("training_status", {}).get("total_metrics_recorded", 0)
                })
                return True
                
            finally:
                Path(model_path).unlink(missing_ok=True)
                
        except Exception as e:
            duration = time.time() - test_start
            self.record_test_result("End-to-End Integration", False, duration, {"error": str(e)})
            return False
            
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 integration tests."""
        
        print("=" * 80)
        print("🚀 PHASE 3 VERL INTEGRATION - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print()
        
        test_functions = [
            ("VERL Integration Components", self.test_verl_integration_components),
            ("Evaluation Framework", self.test_evaluation_framework),
            ("Deployment Infrastructure", self.test_deployment_infrastructure), 
            ("Monitoring & Observability", self.test_monitoring_observability),
            ("End-to-End Integration", self.test_end_to_end_integration)
        ]
        
        total_tests = len(test_functions)
        passed_tests = 0
        
        for test_name, test_func in test_functions:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            
            success = await test_func()
            if success:
                passed_tests += 1
                
        total_duration = time.time() - self.start_time
        
        # Generate final report
        print("\n" + "=" * 80)
        print("📊 PHASE 3 INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["success"] else "❌"
            print(f"{status_icon} {test_name:<40} {result['duration']:>8.2f}s")
            
            if not result["success"] and "error" in result["details"]:
                print(f"    Error: {result['details']['error']}")
                
        print("-" * 80)
        print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Phase 3 integration is working correctly.")
            print("\n✨ Phase 3 Implementation Status: COMPLETE ✨")
        else:
            print("⚠️  Some tests failed. Please review the errors above.")
            
        print("=" * 80)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests,
            "total_duration": total_duration,
            "test_results": self.test_results,
            "overall_success": passed_tests == total_tests
        }
        
    def save_test_report(self, results: Dict[str, Any]) -> None:
        """Save detailed test report to file."""
        
        report_path = Path("phase3_integration_test_report.json")
        
        detailed_report = {
            "test_run_timestamp": time.time(),
            "phase": "Phase 3 - VERL Integration",
            "summary": {
                "total_tests": results["total_tests"],
                "passed_tests": results["passed_tests"],
                "success_rate": results["success_rate"],
                "total_duration": results["total_duration"],
                "overall_success": results["overall_success"]
            },
            "test_results": results["test_results"],
            "environment_info": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2, default=str)
            
        print(f"\n📋 Detailed test report saved to: {report_path}")


async def main():
    """Main test runner function."""
    
    try:
        tester = Phase3IntegrationTester()
        results = await tester.run_all_tests()
        tester.save_test_report(results)
        
        # Exit with appropriate code
        exit_code = 0 if results["overall_success"] else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical error in test runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the integration tests
    asyncio.run(main())