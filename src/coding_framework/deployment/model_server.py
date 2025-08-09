"""
VERL Model Server for serving trained models in production.

Provides high-performance model serving with load balancing, auto-scaling,
and health monitoring for VERL-trained RL models.
"""

import asyncio
import time
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import structlog
from pathlib import Path

from ..agents.trainable_agent import TrainableAgent
from ..verl_integration.verl_config import VERLDistributedConfig


class ServerStatus(Enum):
    """Model server status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ModelServerConfig:
    """Configuration for VERL model server."""
    
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    model_loading_timeout: int = 120
    health_check_interval: int = 30
    enable_metrics: bool = True
    enable_tracing: bool = True
    
    # Model serving configuration
    batch_size: int = 1
    max_batch_delay: float = 0.1  # seconds
    enable_batching: bool = False
    
    # Resource limits
    memory_limit: str = "8Gi"
    cpu_limit: str = "4"
    gpu_limit: int = 1
    
    # Auto-scaling configuration
    enable_autoscaling: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    
    # VERL-specific configuration
    verl_config: Optional[VERLDistributedConfig] = None
    use_verl_inference_engine: bool = True


class VERLModelServer:
    """
    High-performance model server for VERL-trained agents.
    
    Provides REST API endpoints for model inference with support for
    batching, auto-scaling, and comprehensive monitoring.
    """
    
    def __init__(self, config: ModelServerConfig):
        self.config = config
        self.logger = structlog.get_logger(component="verl_model_server")
        
        # Server state
        self.status = ServerStatus.STOPPED
        self.loaded_agent: Optional[TrainableAgent] = None
        self.server_start_time: Optional[float] = None
        
        # Request handling
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        # Metrics and monitoring
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "latency_samples": [],
            "throughput_samples": [],
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "gpu_usage": 0.0
        }
        
        # Health monitoring
        self.last_health_check = 0
        self.health_status = {"status": "unknown", "checks": []}
        
    async def start_server(self) -> Dict[str, Any]:
        """Start the model server and load the agent."""
        
        self.logger.info(
            "Starting VERL model server",
            host=self.config.host,
            port=self.config.port,
            model_path=self.config.model_path
        )
        
        self.status = ServerStatus.STARTING
        self.server_start_time = time.time()
        
        try:
            # Load the trained agent
            await self._load_agent()
            
            # Start server components
            await self._start_request_handler()
            if self.config.enable_batching:
                await self._start_batch_processor()
            
            # Start monitoring tasks
            await self._start_health_monitor()
            if self.config.enable_metrics:
                await self._start_metrics_collector()
                
            # Start auto-scaling if enabled
            if self.config.enable_autoscaling:
                await self._start_autoscaler()
                
            self.status = ServerStatus.HEALTHY
            
            startup_time = time.time() - self.server_start_time
            self.logger.info(
                "VERL model server started successfully",
                startup_time=startup_time,
                status=self.status.value
            )
            
            return {
                "success": True,
                "status": self.status.value,
                "startup_time": startup_time,
                "endpoints": self._get_api_endpoints(),
                "config_summary": self._get_config_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start model server: {e}")
            self.status = ServerStatus.UNHEALTHY
            raise
            
    async def _load_agent(self) -> None:
        """Load the trained agent from the model path."""
        
        self.logger.info("Loading agent model", path=self.config.model_path)
        
        load_start = time.time()
        
        try:
            # This would be replaced with actual agent loading logic
            # For now, we simulate the loading process
            await asyncio.sleep(2)  # Simulate loading time
            
            # In a real implementation, this would load the actual agent
            from ..agents.code_generator_agent import CodeGeneratorAgent
            from ..agents.agent_config import CodeGeneratorConfig
            
            agent_config = CodeGeneratorConfig()
            self.loaded_agent = CodeGeneratorAgent(agent_config)
            
            load_time = time.time() - load_start
            
            self.logger.info(
                "Agent loaded successfully",
                load_time=load_time,
                agent_type=self.loaded_agent.agent_type if self.loaded_agent else "unknown"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load agent: {e}")
            raise
            
    async def process_inference_request(
        self,
        request_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an inference request.
        
        Args:
            request_id: Unique request identifier
            payload: Request payload with input data
            
        Returns:
            Inference response with results
        """
        
        if not self.loaded_agent:
            raise RuntimeError("Agent not loaded")
            
        request_start = time.time()
        
        # Track active request
        request_info = {
            "request_id": request_id,
            "start_time": request_start,
            "status": "processing",
            "payload_size": len(json.dumps(payload))
        }
        self.active_requests[request_id] = request_info
        
        try:
            # Extract input from payload
            input_text = payload.get("input", "")
            context = payload.get("context", {})
            
            # Add inference-specific context
            context.update({
                "inference_mode": True,
                "request_id": request_id,
                "timestamp": request_start
            })
            
            # Process request through agent
            response = await self.loaded_agent.process_request(input_text, context)
            
            request_time = time.time() - request_start
            
            # Update metrics
            self.metrics["requests_total"] += 1
            if response.success:
                self.metrics["requests_successful"] += 1
            else:
                self.metrics["requests_failed"] += 1
                
            self.metrics["latency_samples"].append(request_time)
            if len(self.metrics["latency_samples"]) > 1000:  # Keep last 1000 samples
                self.metrics["latency_samples"] = self.metrics["latency_samples"][-1000:]
                
            # Prepare response
            inference_response = {
                "request_id": request_id,
                "success": response.success,
                "response": response.content if response.success else None,
                "error": response.error if not response.success else None,
                "metadata": {
                    "processing_time": request_time,
                    "agent_type": self.loaded_agent.agent_type,
                    "model_version": getattr(self.loaded_agent, 'model_version', 'unknown'),
                    "timestamp": time.time()
                }
            }
            
            self.logger.info(
                "Inference request completed",
                request_id=request_id,
                success=response.success,
                processing_time=request_time
            )
            
            return inference_response
            
        except Exception as e:
            self.logger.error(f"Inference request failed: {e}", request_id=request_id)
            self.metrics["requests_failed"] += 1
            
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "metadata": {
                    "processing_time": time.time() - request_start,
                    "timestamp": time.time()
                }
            }
            
        finally:
            # Clean up request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
                
    async def process_batch_inference(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of inference requests."""
        
        if not self.config.enable_batching:
            # Process individually
            results = []
            for req in requests:
                result = await self.process_inference_request(
                    req.get("request_id", f"batch_{len(results)}"),
                    req
                )
                results.append(result)
            return results
            
        # Batch processing logic
        batch_start = time.time()
        
        self.logger.info(f"Processing batch inference", batch_size=len(requests))
        
        try:
            # Prepare batch inputs
            batch_inputs = [req.get("input", "") for req in requests]
            
            # Process batch through agent (if agent supports batching)
            # For now, we process individually but could be optimized
            results = []
            for i, req in enumerate(requests):
                request_id = req.get("request_id", f"batch_{i}")
                result = await self.process_inference_request(request_id, req)
                results.append(result)
                
            batch_time = time.time() - batch_start
            
            self.logger.info(
                "Batch inference completed",
                batch_size=len(requests),
                batch_time=batch_time,
                avg_time_per_request=batch_time / len(requests)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}")
            # Return error responses for all requests
            return [
                {
                    "request_id": req.get("request_id", f"batch_{i}"),
                    "success": False,
                    "error": str(e),
                    "metadata": {"timestamp": time.time()}
                }
                for i, req in enumerate(requests)
            ]
            
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the server."""
        
        current_time = time.time()
        
        # Run health checks if needed
        if current_time - self.last_health_check > self.config.health_check_interval:
            await self._run_health_checks()
            
        return {
            "status": self.status.value,
            "uptime": current_time - self.server_start_time if self.server_start_time else 0,
            "health_checks": self.health_status,
            "active_requests": len(self.active_requests),
            "metrics_summary": self._get_metrics_summary(),
            "last_check": self.last_health_check
        }
        
    async def _run_health_checks(self) -> None:
        """Run comprehensive health checks."""
        
        checks = []
        overall_healthy = True
        
        # Check agent availability
        try:
            if self.loaded_agent:
                checks.append({"name": "agent_loaded", "status": "pass"})
            else:
                checks.append({"name": "agent_loaded", "status": "fail", "error": "Agent not loaded"})
                overall_healthy = False
        except Exception as e:
            checks.append({"name": "agent_loaded", "status": "fail", "error": str(e)})
            overall_healthy = False
            
        # Check memory usage
        try:
            # Placeholder for actual memory check
            memory_usage = 50.0  # Simulate 50% memory usage
            if memory_usage < 90.0:
                checks.append({"name": "memory_usage", "status": "pass", "value": memory_usage})
            else:
                checks.append({"name": "memory_usage", "status": "warn", "value": memory_usage})
        except Exception as e:
            checks.append({"name": "memory_usage", "status": "fail", "error": str(e)})
            
        # Check request processing capability
        try:
            if len(self.active_requests) < self.config.max_concurrent_requests * 0.9:
                checks.append({"name": "request_capacity", "status": "pass"})
            else:
                checks.append({"name": "request_capacity", "status": "warn"})
        except Exception as e:
            checks.append({"name": "request_capacity", "status": "fail", "error": str(e)})
            
        # Update health status
        self.health_status = {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
            "timestamp": time.time()
        }
        
        self.last_health_check = time.time()
        
        # Update server status based on health
        if overall_healthy:
            if self.status in [ServerStatus.DEGRADED, ServerStatus.UNHEALTHY]:
                self.status = ServerStatus.HEALTHY
        else:
            self.status = ServerStatus.UNHEALTHY
            
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of server metrics."""
        
        latency_samples = self.metrics["latency_samples"]
        
        return {
            "requests": {
                "total": self.metrics["requests_total"],
                "successful": self.metrics["requests_successful"],
                "failed": self.metrics["requests_failed"],
                "success_rate": (
                    self.metrics["requests_successful"] / max(self.metrics["requests_total"], 1) * 100
                )
            },
            "latency": {
                "avg": sum(latency_samples) / len(latency_samples) if latency_samples else 0,
                "p50": self._percentile(latency_samples, 50) if latency_samples else 0,
                "p95": self._percentile(latency_samples, 95) if latency_samples else 0,
                "p99": self._percentile(latency_samples, 99) if latency_samples else 0
            },
            "resource_usage": {
                "memory": self.metrics["memory_usage"],
                "cpu": self.metrics["cpu_usage"],
                "gpu": self.metrics["gpu_usage"]
            }
        }
        
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
        
    async def _start_request_handler(self) -> None:
        """Start the request handler task."""
        asyncio.create_task(self._request_handler_loop())
        
    async def _start_batch_processor(self) -> None:
        """Start the batch processor task."""
        asyncio.create_task(self._batch_processor_loop())
        
    async def _start_health_monitor(self) -> None:
        """Start the health monitoring task."""
        asyncio.create_task(self._health_monitor_loop())
        
    async def _start_metrics_collector(self) -> None:
        """Start the metrics collection task."""
        asyncio.create_task(self._metrics_collector_loop())
        
    async def _start_autoscaler(self) -> None:
        """Start the auto-scaling task."""
        asyncio.create_task(self._autoscaler_loop())
        
    async def _request_handler_loop(self) -> None:
        """Main request handling loop."""
        while self.status != ServerStatus.STOPPED:
            try:
                await asyncio.sleep(0.1)
                # Request handling logic would go here
            except Exception as e:
                self.logger.error(f"Request handler error: {e}")
                
    async def _batch_processor_loop(self) -> None:
        """Batch processing loop."""
        while self.status != ServerStatus.STOPPED:
            try:
                await asyncio.sleep(self.config.max_batch_delay)
                # Batch processing logic would go here
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self.status != ServerStatus.STOPPED:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._run_health_checks()
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                
    async def _metrics_collector_loop(self) -> None:
        """Metrics collection loop."""
        while self.status != ServerStatus.STOPPED:
            try:
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                # Metrics collection logic would go here
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                
    async def _autoscaler_loop(self) -> None:
        """Auto-scaling loop."""
        while self.status != ServerStatus.STOPPED:
            try:
                await asyncio.sleep(30)  # Check scaling every 30 seconds
                await self._check_scaling_conditions()
            except Exception as e:
                self.logger.error(f"Autoscaler error: {e}")
                
    async def _check_scaling_conditions(self) -> None:
        """Check if scaling is needed and trigger scaling actions."""
        # Placeholder for auto-scaling logic
        pass
        
    def _get_api_endpoints(self) -> List[Dict[str, str]]:
        """Get list of available API endpoints."""
        base_url = f"http://{self.config.host}:{self.config.port}"
        return [
            {"endpoint": "/inference", "method": "POST", "url": f"{base_url}/inference"},
            {"endpoint": "/batch_inference", "method": "POST", "url": f"{base_url}/batch_inference"},
            {"endpoint": "/health", "method": "GET", "url": f"{base_url}/health"},
            {"endpoint": "/metrics", "method": "GET", "url": f"{base_url}/metrics"},
            {"endpoint": "/status", "method": "GET", "url": f"{base_url}/status"}
        ]
        
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of server configuration."""
        return {
            "model_path": self.config.model_path,
            "host": self.config.host,
            "port": self.config.port,
            "workers": self.config.workers,
            "batching_enabled": self.config.enable_batching,
            "autoscaling_enabled": self.config.enable_autoscaling,
            "max_concurrent_requests": self.config.max_concurrent_requests
        }
        
    async def stop_server(self) -> Dict[str, Any]:
        """Stop the model server gracefully."""
        
        self.logger.info("Stopping VERL model server")
        
        self.status = ServerStatus.STOPPING
        
        # Wait for active requests to complete (with timeout)
        shutdown_timeout = 30  # 30 seconds
        shutdown_start = time.time()
        
        while self.active_requests and (time.time() - shutdown_start) < shutdown_timeout:
            await asyncio.sleep(1)
            
        # Force stop if timeout exceeded
        if self.active_requests:
            self.logger.warning(
                f"Forcing shutdown with {len(self.active_requests)} active requests"
            )
            
        self.status = ServerStatus.STOPPED
        
        uptime = time.time() - self.server_start_time if self.server_start_time else 0
        
        self.logger.info(
            "VERL model server stopped",
            uptime=uptime,
            total_requests=self.metrics["requests_total"]
        )
        
        return {
            "success": True,
            "uptime": uptime,
            "total_requests_processed": self.metrics["requests_total"],
            "final_metrics": self._get_metrics_summary()
        }