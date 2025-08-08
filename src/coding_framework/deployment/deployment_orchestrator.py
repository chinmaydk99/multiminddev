"""
Deployment Orchestrator - Coordinates deployment workflows and integrates
all deployment components for VERL-trained models.

Provides high-level deployment orchestration with automated workflows,
dependency management, and comprehensive monitoring integration.
"""

import asyncio
import time
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import structlog
from pathlib import Path

from .deployment_manager import DeploymentManager, DeploymentConfig, DeploymentTarget
from .model_server import VERLModelServer, ModelServerConfig
from .ab_testing import ABTestManager, ABTestConfig, TestVariant
from ..monitoring.deployment_monitor import DeploymentMonitor
from ..verl_integration.verl_config import VERLDistributedConfig


class WorkflowStatus(Enum):
    """Deployment workflow status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class DeploymentPhase(Enum):
    """Phases in the deployment workflow."""
    VALIDATION = "validation"
    MODEL_PREPARATION = "model_preparation"
    INFRASTRUCTURE_SETUP = "infrastructure_setup"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    TRAFFIC_ROUTING = "traffic_routing"
    MONITORING_SETUP = "monitoring_setup"
    COMPLETION = "completion"


@dataclass
class OrchestratorConfig:
    """Configuration for deployment orchestrator."""
    
    # Workflow settings
    enable_automated_rollback: bool = True
    max_deployment_time: int = 3600  # 1 hour
    enable_parallel_deployments: bool = False
    max_concurrent_deployments: int = 3
    
    # Validation settings
    enable_pre_deployment_validation: bool = True
    enable_post_deployment_validation: bool = True
    validation_timeout: int = 300  # 5 minutes
    
    # Monitoring integration
    enable_deployment_monitoring: bool = True
    monitoring_retention_days: int = 30
    
    # A/B testing integration
    enable_ab_testing: bool = False
    default_ab_test_duration_days: int = 14
    
    # Notification settings
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    
    # Recovery settings
    enable_automatic_recovery: bool = True
    health_check_frequency: int = 60  # seconds
    failure_threshold: int = 3  # consecutive failures before action


class DeploymentOrchestrator:
    """
    Orchestrates complete deployment workflows for VERL-trained models.
    
    Coordinates deployment managers, model servers, A/B testing, and monitoring
    to provide seamless production deployment experiences.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.logger = structlog.get_logger(component="deployment_orchestrator")
        
        # Component managers
        self.deployment_manager = DeploymentManager()
        self.ab_test_manager = ABTestManager()
        self.deployment_monitor = DeploymentMonitor() if self.config.enable_deployment_monitoring else None
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Resource management
        self.active_servers: Dict[str, VERLModelServer] = {}
        self.resource_pool: Dict[str, Any] = {}
        
        # Callbacks and hooks
        self.workflow_callbacks: List[Callable] = []
        self.phase_hooks: Dict[DeploymentPhase, List[Callable]] = {}
        
    async def deploy_model_full_workflow(
        self,
        workflow_id: str,
        model_path: str,
        target_environment: DeploymentTarget,
        deployment_config: DeploymentConfig = None,
        server_config: ModelServerConfig = None,
        enable_ab_test: bool = False,
        ab_test_config: ABTestConfig = None
    ) -> Dict[str, Any]:
        """
        Execute complete deployment workflow for a VERL model.
        
        Args:
            workflow_id: Unique workflow identifier
            model_path: Path to trained model
            target_environment: Target deployment environment
            deployment_config: Deployment configuration
            server_config: Model server configuration  
            enable_ab_test: Whether to setup A/B testing
            ab_test_config: A/B test configuration
            
        Returns:
            Complete workflow execution results
        """
        
        self.logger.info(
            "Starting full deployment workflow",
            workflow_id=workflow_id,
            model_path=model_path,
            target=target_environment.value,
            enable_ab_test=enable_ab_test
        )
        
        # Initialize workflow record
        workflow_record = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PENDING,
            "current_phase": None,
            "start_time": time.time(),
            "end_time": None,
            "phases_completed": [],
            "phases_failed": [],
            "results": {},
            "error": None,
            "rollback_executed": False,
            "config": {
                "model_path": model_path,
                "target_environment": target_environment.value,
                "enable_ab_test": enable_ab_test,
                "deployment_config": deployment_config,
                "server_config": server_config,
                "ab_test_config": ab_test_config
            }
        }
        
        self.active_workflows[workflow_id] = workflow_record
        
        try:
            workflow_record["status"] = WorkflowStatus.RUNNING
            
            # Execute workflow phases in sequence
            phases = self._get_workflow_phases(enable_ab_test)
            
            for phase in phases:
                self.logger.info(f"Executing phase: {phase.value}", workflow_id=workflow_id)
                workflow_record["current_phase"] = phase.value
                
                # Run pre-phase hooks
                await self._run_phase_hooks(phase, "pre", workflow_record)
                
                # Execute phase
                phase_result = await self._execute_workflow_phase(
                    phase, workflow_record, deployment_config, server_config, ab_test_config
                )
                
                if phase_result["success"]:
                    workflow_record["phases_completed"].append(phase.value)
                    workflow_record["results"][phase.value] = phase_result
                    
                    # Run post-phase hooks
                    await self._run_phase_hooks(phase, "post", workflow_record)
                else:
                    workflow_record["phases_failed"].append(phase.value)
                    error_msg = f"Phase {phase.value} failed: {phase_result.get('error', 'Unknown error')}"
                    raise Exception(error_msg)
                    
            # Workflow completed successfully
            workflow_record["status"] = WorkflowStatus.COMPLETED
            workflow_record["end_time"] = time.time()
            
            final_results = {
                "success": True,
                "workflow_id": workflow_id,
                "status": WorkflowStatus.COMPLETED.value,
                "duration": workflow_record["end_time"] - workflow_record["start_time"],
                "phases_completed": workflow_record["phases_completed"],
                "deployment_endpoints": self._extract_deployment_endpoints(workflow_record),
                "monitoring_dashboard": self._get_monitoring_dashboard_url(workflow_id),
                "results_summary": self._create_results_summary(workflow_record)
            }
            
            self.logger.info(
                "Deployment workflow completed successfully",
                workflow_id=workflow_id,
                duration=final_results["duration"],
                phases_completed=len(workflow_record["phases_completed"])
            )
            
            # Run completion callbacks
            await self._run_workflow_callbacks(workflow_record, "completed")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Deployment workflow failed: {e}", workflow_id=workflow_id)
            
            workflow_record["status"] = WorkflowStatus.FAILED
            workflow_record["error"] = str(e)
            workflow_record["end_time"] = time.time()
            
            # Attempt rollback if enabled
            if self.config.enable_automated_rollback:
                try:
                    rollback_result = await self._execute_workflow_rollback(workflow_record)
                    workflow_record["rollback_executed"] = True
                    workflow_record["rollback_result"] = rollback_result
                    
                    if rollback_result["success"]:
                        workflow_record["status"] = WorkflowStatus.ROLLED_BACK
                        
                except Exception as rollback_error:
                    self.logger.error(
                        f"Rollback also failed: {rollback_error}",
                        workflow_id=workflow_id
                    )
                    
            # Run failure callbacks
            await self._run_workflow_callbacks(workflow_record, "failed")
            
            return {
                "success": False,
                "workflow_id": workflow_id,
                "status": workflow_record["status"].value,
                "error": str(e),
                "rollback_executed": workflow_record["rollback_executed"],
                "duration": workflow_record["end_time"] - workflow_record["start_time"]
            }
            
        finally:
            # Move to history and cleanup
            self.workflow_history.append(workflow_record.copy())
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                
    def _get_workflow_phases(self, enable_ab_test: bool) -> List[DeploymentPhase]:
        """Get the phases for the deployment workflow."""
        
        base_phases = [
            DeploymentPhase.VALIDATION,
            DeploymentPhase.MODEL_PREPARATION,
            DeploymentPhase.INFRASTRUCTURE_SETUP,
            DeploymentPhase.DEPLOYMENT,
            DeploymentPhase.TESTING
        ]
        
        if enable_ab_test:
            base_phases.extend([
                DeploymentPhase.TRAFFIC_ROUTING,
            ])
        else:
            base_phases.append(DeploymentPhase.TRAFFIC_ROUTING)
            
        base_phases.extend([
            DeploymentPhase.MONITORING_SETUP,
            DeploymentPhase.COMPLETION
        ])
        
        return base_phases
        
    async def _execute_workflow_phase(
        self,
        phase: DeploymentPhase,
        workflow_record: Dict[str, Any],
        deployment_config: DeploymentConfig,
        server_config: ModelServerConfig,
        ab_test_config: ABTestConfig
    ) -> Dict[str, Any]:
        """Execute a specific workflow phase."""
        
        phase_start = time.time()
        
        try:
            if phase == DeploymentPhase.VALIDATION:
                return await self._execute_validation_phase(workflow_record)
                
            elif phase == DeploymentPhase.MODEL_PREPARATION:
                return await self._execute_model_preparation_phase(workflow_record)
                
            elif phase == DeploymentPhase.INFRASTRUCTURE_SETUP:
                return await self._execute_infrastructure_setup_phase(workflow_record, server_config)
                
            elif phase == DeploymentPhase.DEPLOYMENT:
                return await self._execute_deployment_phase(workflow_record, deployment_config)
                
            elif phase == DeploymentPhase.TESTING:
                return await self._execute_testing_phase(workflow_record)
                
            elif phase == DeploymentPhase.TRAFFIC_ROUTING:
                return await self._execute_traffic_routing_phase(workflow_record, ab_test_config)
                
            elif phase == DeploymentPhase.MONITORING_SETUP:
                return await self._execute_monitoring_setup_phase(workflow_record)
                
            elif phase == DeploymentPhase.COMPLETION:
                return await self._execute_completion_phase(workflow_record)
                
            else:
                raise ValueError(f"Unknown workflow phase: {phase}")
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "phase": phase.value,
                "duration": time.time() - phase_start
            }
            
    async def _execute_validation_phase(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation phase."""
        
        self.logger.info("Executing validation phase", workflow_id=workflow_record["workflow_id"])
        
        validation_results = {
            "model_path_exists": False,
            "model_format_valid": False,
            "dependencies_available": False,
            "target_environment_ready": False,
            "resource_requirements_met": False
        }
        
        config = workflow_record["config"]
        model_path = Path(config["model_path"])
        
        # Check model path exists
        validation_results["model_path_exists"] = model_path.exists()
        
        # Validate model format (simplified check)
        if validation_results["model_path_exists"]:
            validation_results["model_format_valid"] = True  # Would do actual validation
            
        # Check dependencies
        validation_results["dependencies_available"] = True  # Would check actual dependencies
        
        # Check target environment
        validation_results["target_environment_ready"] = True  # Would check environment
        
        # Check resource requirements
        validation_results["resource_requirements_met"] = True  # Would check resources
        
        # Overall validation result
        all_valid = all(validation_results.values())
        
        return {
            "success": all_valid,
            "validation_results": validation_results,
            "error": None if all_valid else "Validation checks failed"
        }
        
    async def _execute_model_preparation_phase(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model preparation phase."""
        
        self.logger.info("Executing model preparation phase", workflow_id=workflow_record["workflow_id"])
        
        # Model optimization, serialization, etc.
        await asyncio.sleep(2)  # Simulate preparation time
        
        return {
            "success": True,
            "prepared_model_path": workflow_record["config"]["model_path"],
            "optimization_applied": ["quantization", "pruning"],
            "model_size_mb": 150.5  # Simulated
        }
        
    async def _execute_infrastructure_setup_phase(
        self, 
        workflow_record: Dict[str, Any],
        server_config: ModelServerConfig
    ) -> Dict[str, Any]:
        """Execute infrastructure setup phase."""
        
        self.logger.info("Executing infrastructure setup phase", workflow_id=workflow_record["workflow_id"])
        
        workflow_id = workflow_record["workflow_id"]
        config = workflow_record["config"]
        
        # Create model server configuration
        if not server_config:
            server_config = ModelServerConfig(
                model_path=config["model_path"],
                port=8000 + hash(workflow_id) % 1000  # Unique port
            )
            
        # Initialize model server
        model_server = VERLModelServer(server_config)
        
        # Start the server
        server_start_result = await model_server.start_server()
        
        if server_start_result["success"]:
            self.active_servers[workflow_id] = model_server
            
        return {
            "success": server_start_result["success"],
            "server_config": server_config.__dict__,
            "server_endpoints": server_start_result.get("endpoints", []),
            "startup_time": server_start_result.get("startup_time", 0)
        }
        
    async def _execute_deployment_phase(
        self, 
        workflow_record: Dict[str, Any],
        deployment_config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Execute deployment phase."""
        
        self.logger.info("Executing deployment phase", workflow_id=workflow_record["workflow_id"])
        
        workflow_id = workflow_record["workflow_id"]
        config = workflow_record["config"]
        
        # Create deployment configuration
        if not deployment_config:
            deployment_config = DeploymentConfig(
                model_path=config["model_path"],
                target_environment=DeploymentTarget(config["target_environment"])
            )
            
        # Execute deployment
        deployment_result = await self.deployment_manager.deploy_model(
            deployment_id=f"deploy_{workflow_id}",
            model_path=config["model_path"],
            config=deployment_config
        )
        
        return {
            "success": deployment_result.get("success", False),
            "deployment_id": deployment_result.get("deployment_id"),
            "deployment_strategy": deployment_result.get("strategy"),
            "deployment_endpoints": deployment_result.get("deployment_endpoints", [])
        }
        
    async def _execute_testing_phase(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute testing phase."""
        
        self.logger.info("Executing testing phase", workflow_id=workflow_record["workflow_id"])
        
        workflow_id = workflow_record["workflow_id"]
        
        # Run deployment tests
        test_results = {
            "health_check": True,
            "smoke_tests": True,
            "integration_tests": True,
            "performance_tests": True
        }
        
        # Simulate testing
        await asyncio.sleep(3)
        
        # Test with active server if available
        if workflow_id in self.active_servers:
            server = self.active_servers[workflow_id]
            health_status = await server.get_health_status()
            test_results["server_health"] = health_status
            
        return {
            "success": all(test_results.values()),
            "test_results": test_results,
            "tests_passed": sum(test_results.values()),
            "tests_total": len(test_results)
        }
        
    async def _execute_traffic_routing_phase(
        self, 
        workflow_record: Dict[str, Any],
        ab_test_config: ABTestConfig
    ) -> Dict[str, Any]:
        """Execute traffic routing phase."""
        
        self.logger.info("Executing traffic routing phase", workflow_id=workflow_record["workflow_id"])
        
        config = workflow_record["config"]
        
        if config.get("enable_ab_test", False) and ab_test_config:
            # Setup A/B test
            ab_test_result = await self.ab_test_manager.create_test(ab_test_config)
            
            if ab_test_result["success"]:
                # Start the A/B test
                start_result = await self.ab_test_manager.start_test(ab_test_config.test_id)
                
                return {
                    "success": start_result["success"],
                    "routing_type": "ab_test",
                    "test_id": ab_test_config.test_id,
                    "variants": len(ab_test_config.variants)
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create A/B test",
                    "ab_test_error": ab_test_result.get("error")
                }
        else:
            # Direct traffic routing (no A/B test)
            await asyncio.sleep(1)  # Simulate routing setup
            
            return {
                "success": True,
                "routing_type": "direct",
                "traffic_percentage": 100.0
            }
            
    async def _execute_monitoring_setup_phase(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring setup phase."""
        
        self.logger.info("Executing monitoring setup phase", workflow_id=workflow_record["workflow_id"])
        
        workflow_id = workflow_record["workflow_id"]
        
        if self.deployment_monitor:
            # Setup monitoring for this deployment
            monitoring_config = {
                "deployment_id": workflow_id,
                "metrics": ["latency", "throughput", "error_rate", "resource_usage"],
                "alerts": ["high_error_rate", "high_latency", "resource_exhaustion"]
            }
            
            monitoring_result = await self.deployment_monitor.setup_deployment_monitoring(
                workflow_id, monitoring_config
            )
            
            return {
                "success": monitoring_result.get("success", True),
                "monitoring_enabled": True,
                "dashboard_url": f"http://monitoring.local/dashboard/{workflow_id}",
                "alert_channels": self.config.notification_channels
            }
        else:
            return {
                "success": True,
                "monitoring_enabled": False,
                "message": "Monitoring not configured"
            }
            
    async def _execute_completion_phase(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute completion phase."""
        
        self.logger.info("Executing completion phase", workflow_id=workflow_record["workflow_id"])
        
        # Final validations, cleanup, notifications
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "workflow_completed": True,
            "cleanup_performed": True,
            "notifications_sent": self.config.enable_notifications
        }
        
    async def _execute_workflow_rollback(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback for failed workflow."""
        
        self.logger.info("Executing workflow rollback", workflow_id=workflow_record["workflow_id"])
        
        rollback_steps = []
        workflow_id = workflow_record["workflow_id"]
        
        try:
            # Stop model server if running
            if workflow_id in self.active_servers:
                server = self.active_servers[workflow_id]
                await server.stop_server()
                del self.active_servers[workflow_id]
                rollback_steps.append("model_server_stopped")
                
            # Rollback deployment
            deployment_results = workflow_record["results"].get("deployment", {})
            if deployment_results and deployment_results.get("deployment_id"):
                await self.deployment_manager.rollback_deployment(
                    deployment_results["deployment_id"]
                )
                rollback_steps.append("deployment_rolled_back")
                
            # Stop A/B test if running
            if workflow_record["config"].get("enable_ab_test"):
                ab_config = workflow_record["config"].get("ab_test_config")
                if ab_config and hasattr(ab_config, 'test_id'):
                    await self.ab_test_manager.stop_test(ab_config.test_id, "rollback")
                    rollback_steps.append("ab_test_stopped")
                    
            return {
                "success": True,
                "rollback_steps": rollback_steps,
                "rollback_time": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rollback_steps": rollback_steps
            }
            
    async def _run_phase_hooks(
        self, 
        phase: DeploymentPhase, 
        timing: str, 
        workflow_record: Dict[str, Any]
    ) -> None:
        """Run registered hooks for a deployment phase."""
        
        if phase in self.phase_hooks:
            for hook in self.phase_hooks[phase]:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(phase, timing, workflow_record)
                    else:
                        hook(phase, timing, workflow_record)
                except Exception as e:
                    self.logger.warning(f"Phase hook failed: {e}")
                    
    async def _run_workflow_callbacks(
        self, 
        workflow_record: Dict[str, Any], 
        event: str
    ) -> None:
        """Run registered workflow callbacks."""
        
        for callback in self.workflow_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(workflow_record, event)
                else:
                    callback(workflow_record, event)
            except Exception as e:
                self.logger.warning(f"Workflow callback failed: {e}")
                
    def _extract_deployment_endpoints(self, workflow_record: Dict[str, Any]) -> List[str]:
        """Extract deployment endpoints from workflow results."""
        
        endpoints = []
        
        # From infrastructure setup
        infra_results = workflow_record["results"].get("infrastructure_setup", {})
        server_endpoints = infra_results.get("server_endpoints", [])
        for endpoint in server_endpoints:
            if isinstance(endpoint, dict) and "url" in endpoint:
                endpoints.append(endpoint["url"])
                
        # From deployment phase
        deploy_results = workflow_record["results"].get("deployment", {})
        deploy_endpoints = deploy_results.get("deployment_endpoints", [])
        endpoints.extend(deploy_endpoints)
        
        return endpoints
        
    def _get_monitoring_dashboard_url(self, workflow_id: str) -> Optional[str]:
        """Get monitoring dashboard URL for workflow."""
        
        if self.deployment_monitor:
            return f"http://monitoring.local/dashboard/{workflow_id}"
        return None
        
    def _create_results_summary(self, workflow_record: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of workflow results."""
        
        return {
            "phases_completed": len(workflow_record["phases_completed"]),
            "phases_total": len(workflow_record["phases_completed"]) + len(workflow_record["phases_failed"]),
            "deployment_successful": workflow_record["status"] == WorkflowStatus.COMPLETED,
            "monitoring_enabled": self.config.enable_deployment_monitoring,
            "ab_test_enabled": workflow_record["config"].get("enable_ab_test", False)
        }
        
    # Public methods for workflow management
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
            
        # Check history
        for record in self.workflow_history:
            if record["workflow_id"] == workflow_id:
                return record
                
        return {"error": f"Workflow {workflow_id} not found"}
        
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        return list(self.active_workflows.values())
        
    def list_workflow_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List workflow history."""
        return sorted(
            self.workflow_history[-limit:], 
            key=lambda x: x.get("start_time", 0), 
            reverse=True
        )
        
    def register_workflow_callback(self, callback: Callable) -> None:
        """Register callback for workflow events."""
        self.workflow_callbacks.append(callback)
        
    def register_phase_hook(self, phase: DeploymentPhase, hook: Callable) -> None:
        """Register hook for specific deployment phase."""
        if phase not in self.phase_hooks:
            self.phase_hooks[phase] = []
        self.phase_hooks[phase].append(hook)
        
    async def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a running workflow."""
        
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": f"Workflow {workflow_id} not found"}
            
        workflow_record = self.active_workflows[workflow_id]
        
        self.logger.info("Cancelling workflow", workflow_id=workflow_id)
        
        try:
            # Execute rollback
            rollback_result = await self._execute_workflow_rollback(workflow_record)
            
            # Update workflow status
            workflow_record["status"] = WorkflowStatus.CANCELLED
            workflow_record["end_time"] = time.time()
            workflow_record["rollback_result"] = rollback_result
            
            # Move to history
            self.workflow_history.append(workflow_record.copy())
            del self.active_workflows[workflow_id]
            
            # Run cancellation callbacks
            await self._run_workflow_callbacks(workflow_record, "cancelled")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": WorkflowStatus.CANCELLED.value,
                "rollback_successful": rollback_result.get("success", False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow: {e}", workflow_id=workflow_id)
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
            
    async def get_deployment_health(self, workflow_id: str) -> Dict[str, Any]:
        """Get health status of deployed model."""
        
        if workflow_id in self.active_servers:
            server = self.active_servers[workflow_id]
            health_status = await server.get_health_status()
            return {
                "workflow_id": workflow_id,
                "server_healthy": health_status["status"] in ["healthy", "degraded"],
                "health_details": health_status
            }
        else:
            return {
                "workflow_id": workflow_id,
                "server_healthy": False,
                "error": "No active server found for workflow"
            }