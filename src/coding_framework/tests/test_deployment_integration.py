"""
Comprehensive integration tests for the deployment infrastructure.

Tests the full deployment workflow including deployment manager, model server,
A/B testing, and orchestration capabilities.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from ..deployment.deployment_manager import (
    DeploymentManager, 
    DeploymentConfig, 
    DeploymentTarget, 
    DeploymentStatus
)
from ..deployment.model_server import VERLModelServer, ModelServerConfig, ServerStatus
from ..deployment.ab_testing import ABTestManager, ABTestConfig, TestVariant, TestStatus
from ..deployment.deployment_orchestrator import (
    DeploymentOrchestrator, 
    OrchestratorConfig, 
    WorkflowStatus
)
from ..agents.code_generator_agent import CodeGeneratorAgent
from ..agents.agent_config import CodeGeneratorConfig


class TestDeploymentManager:
    """Test cases for the deployment manager."""
    
    @pytest.fixture
    async def deployment_manager(self):
        """Create deployment manager for testing."""
        return DeploymentManager()
        
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration for testing."""
        return DeploymentConfig(
            model_path="/tmp/test_model",
            target_environment=DeploymentTarget.DEVELOPMENT,
            replicas=2,
            rollback_on_failure=True
        )
        
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, deployment_manager, deployment_config):
        """Test blue-green deployment strategy."""
        
        deployment_config.deployment_strategy = "blue_green"
        
        result = await deployment_manager.deploy_model(
            deployment_id="test_bg_deploy",
            model_path="/tmp/test_model",
            config=deployment_config
        )
        
        assert result["success"] == True
        assert result["strategy"] == "blue_green"
        assert "steps_completed" in result
        assert "traffic_switched" in result["steps_completed"]
        
    @pytest.mark.asyncio
    async def test_rolling_deployment(self, deployment_manager, deployment_config):
        """Test rolling deployment strategy."""
        
        deployment_config.deployment_strategy = "rolling"
        
        result = await deployment_manager.deploy_model(
            deployment_id="test_rolling_deploy", 
            model_path="/tmp/test_model",
            config=deployment_config
        )
        
        assert result["success"] == True
        assert result["strategy"] == "rolling"
        assert result["total_replicas"] == deployment_config.replicas
        
    @pytest.mark.asyncio
    async def test_canary_deployment(self, deployment_manager, deployment_config):
        """Test canary deployment strategy."""
        
        deployment_config.deployment_strategy = "canary"
        deployment_config.canary_traffic_percentage = 10.0
        
        result = await deployment_manager.deploy_model(
            deployment_id="test_canary_deploy",
            model_path="/tmp/test_model", 
            config=deployment_config
        )
        
        assert result["success"] == True
        assert result["strategy"] == "canary"
        assert "canary_promoted" in result["steps_completed"]
        
    @pytest.mark.asyncio
    async def test_deployment_rollback(self, deployment_manager, deployment_config):
        """Test deployment rollback functionality."""
        
        # First deploy successfully
        await deployment_manager.deploy_model(
            deployment_id="test_rollback_deploy",
            model_path="/tmp/test_model",
            config=deployment_config
        )
        
        # Then rollback
        rollback_result = await deployment_manager.rollback_deployment("test_rollback_deploy")
        
        assert rollback_result["success"] == True
        assert "health_status" in rollback_result
        
    def test_deployment_status_tracking(self, deployment_manager):
        """Test deployment status tracking."""
        
        # Initially no deployments
        deployments = deployment_manager.list_deployments()
        assert len(deployments) == 0
        
        # Check status retrieval
        status = deployment_manager.get_deployment_status("nonexistent")
        assert "error" in status


class TestVERLModelServer:
    """Test cases for the VERL model server."""
    
    @pytest.fixture
    def server_config(self):
        """Create server configuration for testing."""
        return ModelServerConfig(
            model_path="/tmp/test_model",
            host="localhost",
            port=8080,
            max_concurrent_requests=50
        )
        
    @pytest.fixture
    async def model_server(self, server_config):
        """Create model server for testing."""
        return VERLModelServer(server_config)
        
    @pytest.mark.asyncio
    async def test_server_startup_and_shutdown(self, model_server):
        """Test server startup and shutdown process."""
        
        # Start server
        start_result = await model_server.start_server()
        
        assert start_result["success"] == True
        assert model_server.status == ServerStatus.HEALTHY
        assert "endpoints" in start_result
        
        # Stop server
        stop_result = await model_server.stop_server()
        
        assert stop_result["success"] == True
        assert model_server.status == ServerStatus.STOPPED
        
    @pytest.mark.asyncio
    async def test_inference_request_processing(self, model_server):
        """Test inference request processing."""
        
        # Start server first
        await model_server.start_server()
        
        # Process inference request
        test_payload = {
            "input": "def fibonacci(n):",
            "context": {"max_tokens": 100}
        }
        
        result = await model_server.process_inference_request(
            request_id="test_request_001",
            payload=test_payload
        )
        
        assert "request_id" in result
        assert "success" in result
        assert "metadata" in result
        
        # Stop server
        await model_server.stop_server()
        
    @pytest.mark.asyncio
    async def test_batch_inference_processing(self, model_server):
        """Test batch inference processing."""
        
        # Start server
        await model_server.start_server()
        
        # Prepare batch requests
        batch_requests = [
            {"request_id": "batch_req_1", "input": "def add(a, b):"},
            {"request_id": "batch_req_2", "input": "def multiply(x, y):"},
            {"request_id": "batch_req_3", "input": "def divide(a, b):"}
        ]
        
        # Process batch
        results = await model_server.process_batch_inference(batch_requests)
        
        assert len(results) == 3
        for result in results:
            assert "request_id" in result
            assert "success" in result
            
        # Stop server
        await model_server.stop_server()
        
    @pytest.mark.asyncio
    async def test_health_monitoring(self, model_server):
        """Test server health monitoring."""
        
        # Start server
        await model_server.start_server()
        
        # Get health status
        health_status = await model_server.get_health_status()
        
        assert "status" in health_status
        assert "uptime" in health_status
        assert "health_checks" in health_status
        assert "metrics_summary" in health_status
        
        # Stop server
        await model_server.stop_server()


class TestABTestManager:
    """Test cases for the A/B testing manager."""
    
    @pytest.fixture
    def ab_test_config(self):
        """Create A/B test configuration for testing."""
        variants = [
            TestVariant(
                variant_id="control",
                name="Control Model",
                model_path="/tmp/control_model",
                traffic_allocation=50.0
            ),
            TestVariant(
                variant_id="experiment", 
                name="Experimental Model",
                model_path="/tmp/experiment_model",
                traffic_allocation=50.0
            )
        ]
        
        return ABTestConfig(
            test_id="test_ab_001",
            test_name="Code Generation Model Comparison",
            description="Comparing control vs experimental model",
            variants=variants,
            duration_days=7,
            min_sample_size=100
        )
        
    @pytest.fixture
    def ab_test_manager(self):
        """Create A/B test manager for testing."""
        return ABTestManager()
        
    @pytest.mark.asyncio
    async def test_ab_test_creation_and_startup(self, ab_test_manager, ab_test_config):
        """Test A/B test creation and startup."""
        
        # Create test
        create_result = await ab_test_manager.create_test(ab_test_config)
        
        assert create_result["success"] == True
        assert create_result["test_id"] == ab_test_config.test_id
        assert create_result["variants_deployed"] == 2
        
        # Start test
        start_result = await ab_test_manager.start_test(ab_test_config.test_id)
        
        assert start_result["success"] == True
        assert start_result["status"] == TestStatus.RUNNING.value
        
    @pytest.mark.asyncio
    async def test_request_routing(self, ab_test_manager, ab_test_config):
        """Test request routing in A/B test."""
        
        # Create and start test
        await ab_test_manager.create_test(ab_test_config)
        await ab_test_manager.start_test(ab_test_config.test_id)
        
        # Route requests
        routing_results = []
        for i in range(100):
            result = await ab_test_manager.route_request(
                test_id=ab_test_config.test_id,
                request_context={"user_id": f"user_{i}"}
            )
            routing_results.append(result["variant_id"])
            
        # Check that both variants received traffic
        unique_variants = set(routing_results)
        assert len(unique_variants) == 2
        assert "control" in unique_variants
        assert "experiment" in unique_variants
        
    @pytest.mark.asyncio
    async def test_result_recording_and_analysis(self, ab_test_manager, ab_test_config):
        """Test result recording and statistical analysis."""
        
        # Create and start test
        await ab_test_manager.create_test(ab_test_config)
        await ab_test_manager.start_test(ab_test_config.test_id)
        
        # Record some results for both variants
        for i in range(50):
            # Control variant results
            await ab_test_manager.record_result(
                test_id=ab_test_config.test_id,
                variant_id="control",
                result={
                    "success": True,
                    "latency": 0.1 + (i % 10) * 0.01,  # Variable latency
                    "conversion": i % 3 == 0  # 33% conversion rate
                }
            )
            
            # Experiment variant results (slightly better)
            await ab_test_manager.record_result(
                test_id=ab_test_config.test_id,
                variant_id="experiment", 
                result={
                    "success": True,
                    "latency": 0.08 + (i % 10) * 0.01,  # Slightly better latency
                    "conversion": i % 2 == 0  # 50% conversion rate
                }
            )
            
        # Get test results
        test_results = await ab_test_manager.get_test_results(ab_test_config.test_id)
        
        assert test_results["test_id"] == ab_test_config.test_id
        assert len(test_results["variants"]) == 2
        assert "statistical_analysis" in test_results
        
        # Check variant metrics
        for variant in test_results["variants"]:
            assert variant["total_requests"] == 50
            assert variant["conversion_rate"] > 0
            assert variant["avg_latency"] > 0
            
    @pytest.mark.asyncio
    async def test_ab_test_stopping(self, ab_test_manager, ab_test_config):
        """Test A/B test stopping functionality."""
        
        # Create and start test
        await ab_test_manager.create_test(ab_test_config)
        await ab_test_manager.start_test(ab_test_config.test_id)
        
        # Stop test
        stop_result = await ab_test_manager.stop_test(
            ab_test_config.test_id, 
            reason="manual_stop"
        )
        
        assert stop_result["success"] == True
        assert stop_result["test_id"] == ab_test_config.test_id
        assert "final_results" in stop_result


class TestDeploymentOrchestrator:
    """Test cases for the deployment orchestrator."""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Create orchestrator configuration for testing."""
        return OrchestratorConfig(
            enable_automated_rollback=True,
            max_deployment_time=1800,
            enable_pre_deployment_validation=True,
            enable_deployment_monitoring=False  # Simplified for testing
        )
        
    @pytest.fixture
    def deployment_orchestrator(self, orchestrator_config):
        """Create deployment orchestrator for testing."""
        return DeploymentOrchestrator(orchestrator_config)
        
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration for testing."""
        return DeploymentConfig(
            model_path="/tmp/test_model",
            target_environment=DeploymentTarget.DEVELOPMENT,
            deployment_strategy="blue_green"
        )
        
    @pytest.fixture
    def server_config(self):
        """Create server configuration for testing.""" 
        return ModelServerConfig(
            model_path="/tmp/test_model",
            port=8090
        )
        
    @pytest.mark.asyncio
    async def test_full_deployment_workflow(
        self, 
        deployment_orchestrator, 
        deployment_config, 
        server_config
    ):
        """Test complete deployment workflow execution."""
        
        result = await deployment_orchestrator.deploy_model_full_workflow(
            workflow_id="test_workflow_001",
            model_path="/tmp/test_model",
            target_environment=DeploymentTarget.DEVELOPMENT,
            deployment_config=deployment_config,
            server_config=server_config,
            enable_ab_test=False
        )
        
        assert result["success"] == True
        assert result["workflow_id"] == "test_workflow_001"
        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert "deployment_endpoints" in result
        assert "results_summary" in result
        assert len(result["phases_completed"]) > 0
        
    @pytest.mark.asyncio 
    async def test_workflow_with_ab_testing(
        self,
        deployment_orchestrator,
        deployment_config,
        server_config
    ):
        """Test deployment workflow with A/B testing enabled."""
        
        # Create A/B test configuration
        variants = [
            TestVariant(
                variant_id="current",
                name="Current Model", 
                model_path="/tmp/current_model",
                traffic_allocation=50.0
            ),
            TestVariant(
                variant_id="new",
                name="New Model",
                model_path="/tmp/new_model",
                traffic_allocation=50.0
            )
        ]
        
        ab_test_config = ABTestConfig(
            test_id="workflow_ab_test",
            test_name="Deployment A/B Test",
            description="Testing new model deployment",
            variants=variants,
            duration_days=1  # Short test for validation
        )
        
        result = await deployment_orchestrator.deploy_model_full_workflow(
            workflow_id="test_workflow_ab_002",
            model_path="/tmp/test_model",
            target_environment=DeploymentTarget.STAGING,
            deployment_config=deployment_config,
            server_config=server_config,
            enable_ab_test=True,
            ab_test_config=ab_test_config
        )
        
        assert result["success"] == True
        assert result["results_summary"]["ab_test_enabled"] == True
        
    @pytest.mark.asyncio
    async def test_workflow_failure_and_rollback(
        self,
        deployment_orchestrator,
        deployment_config,
        server_config
    ):
        """Test workflow failure handling and rollback."""
        
        # Simulate failure by using invalid model path
        deployment_config.model_path = "/nonexistent/model/path"
        
        result = await deployment_orchestrator.deploy_model_full_workflow(
            workflow_id="test_workflow_failure_003", 
            model_path="/nonexistent/model/path",
            target_environment=DeploymentTarget.DEVELOPMENT,
            deployment_config=deployment_config,
            server_config=server_config
        )
        
        # Should fail but attempt rollback
        assert result["success"] == False
        assert "error" in result
        assert result["rollback_executed"] == True
        
    @pytest.mark.asyncio
    async def test_workflow_cancellation(
        self,
        deployment_orchestrator,
        deployment_config,
        server_config
    ):
        """Test workflow cancellation."""
        
        # Start a workflow
        workflow_task = asyncio.create_task(
            deployment_orchestrator.deploy_model_full_workflow(
                workflow_id="test_workflow_cancel_004",
                model_path="/tmp/test_model", 
                target_environment=DeploymentTarget.DEVELOPMENT,
                deployment_config=deployment_config,
                server_config=server_config
            )
        )
        
        # Let it start, then cancel
        await asyncio.sleep(0.1)  # Brief delay to let workflow start
        
        cancel_result = await deployment_orchestrator.cancel_workflow("test_workflow_cancel_004")
        
        # Cancel the task
        workflow_task.cancel()
        
        assert cancel_result["success"] == True
        assert cancel_result["status"] == WorkflowStatus.CANCELLED.value
        
    def test_workflow_status_tracking(self, deployment_orchestrator):
        """Test workflow status tracking and history."""
        
        # Initially no workflows
        active_workflows = deployment_orchestrator.list_active_workflows()
        assert len(active_workflows) == 0
        
        history = deployment_orchestrator.list_workflow_history()
        assert len(history) == 0
        
        # Test status retrieval
        status = deployment_orchestrator.get_workflow_status("nonexistent")
        assert "error" in status


@pytest.mark.integration
class TestFullDeploymentIntegration:
    """Integration tests for the complete deployment pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_workflow(self):
        """Test complete end-to-end deployment workflow."""
        
        # Create a temporary model file for testing
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            model_path = temp_file.name
            # Write dummy model data
            temp_file.write(b"dummy_model_data")
            
        try:
            # Setup components
            orchestrator_config = OrchestratorConfig(
                enable_deployment_monitoring=False,  # Simplified for testing
                max_deployment_time=300  # 5 minutes for testing
            )
            orchestrator = DeploymentOrchestrator(orchestrator_config)
            
            deployment_config = DeploymentConfig(
                model_path=model_path,
                target_environment=DeploymentTarget.DEVELOPMENT,
                deployment_strategy="blue_green",
                replicas=1  # Single replica for testing
            )
            
            server_config = ModelServerConfig(
                model_path=model_path,
                port=8095,  # Unique port for testing
                workers=1
            )
            
            # Execute full workflow
            result = await orchestrator.deploy_model_full_workflow(
                workflow_id="integration_test_workflow",
                model_path=model_path,
                target_environment=DeploymentTarget.DEVELOPMENT,
                deployment_config=deployment_config,
                server_config=server_config
            )
            
            # Verify successful deployment
            assert result["success"] == True
            assert result["status"] == WorkflowStatus.COMPLETED.value
            assert len(result["phases_completed"]) > 5  # Multiple phases completed
            
            # Test deployment health check
            health_result = await orchestrator.get_deployment_health("integration_test_workflow")
            assert "workflow_id" in health_result
            
        finally:
            # Cleanup temporary file
            Path(model_path).unlink(missing_ok=True)
            
    @pytest.mark.asyncio
    async def test_deployment_with_monitoring_integration(self):
        """Test deployment with monitoring integration enabled."""
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            model_path = temp_file.name
            temp_file.write(b"dummy_model_data")
            
        try:
            # Enable monitoring
            orchestrator_config = OrchestratorConfig(
                enable_deployment_monitoring=True,
                monitoring_retention_days=7
            )
            orchestrator = DeploymentOrchestrator(orchestrator_config)
            
            deployment_config = DeploymentConfig(
                model_path=model_path,
                target_environment=DeploymentTarget.STAGING,
                enable_monitoring=True
            )
            
            server_config = ModelServerConfig(
                model_path=model_path,
                port=8096,
                enable_metrics=True
            )
            
            result = await orchestrator.deploy_model_full_workflow(
                workflow_id="monitoring_integration_test",
                model_path=model_path,
                target_environment=DeploymentTarget.STAGING,
                deployment_config=deployment_config,
                server_config=server_config
            )
            
            assert result["success"] == True
            assert "monitoring_dashboard" in result
            assert result["results_summary"]["monitoring_enabled"] == True
            
        finally:
            Path(model_path).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])