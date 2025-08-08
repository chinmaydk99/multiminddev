# Architecture Patterns and Conventions

## Agent Pattern
All agents inherit from `BaseAgent` (src/coding_framework/agents/base_agent.py):

### Standard Agent Structure
```python
class BaseAgent(ABC):
    # Core properties
    @property
    @abstractmethod
    def agent_type(self) -> str: pass
    
    @property  
    @abstractmethod
    def system_prompt(self) -> str: pass
    
    # Main processing method
    @abstractmethod
    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> AgentResponse
```

### Agent Response Format
All agents return standardized `AgentResponse`:
- `agent_type`: str
- `success`: bool  
- `content`: str
- `metadata`: Dict[str, Any]
- `execution_time`: float
- `timestamp`: float
- `error`: Optional[str]

## Supervisor Pattern
`CodingSupervisor` orchestrates agents:
- Initializes LLM interface, agents, and workflows
- Provides health checks for all components
- Handles performance metrics tracking
- Contains placeholder `train_agents()` method (Phase 2 target)

## Configuration Pattern
Hierarchical YAML configuration with Pydantic validation:
- `Config` class with nested agent configs
- Environment variable overrides supported
- Separate configs for different environments (Colab, production)

## Logging Pattern
Structured logging with `structlog`:
- Agent-specific loggers with context
- Performance metrics integration
- Rich console output support

## Error Handling Pattern
- Fail-fast principle
- Comprehensive exception handling with structured logging
- Health check methods throughout system
- Graceful degradation where possible

## Testing Pattern
- Tests live next to code they test
- Pytest with async support
- Unit, integration, and e2e test categories
- Coverage reporting with HTML output

## Current Gaps (Phase 2 Targets)
- `src/coding_framework/training/` directory is empty
- `train_agents()` method is placeholder only
- No reward function implementations
- No VERL pipeline integration
- No training data handling