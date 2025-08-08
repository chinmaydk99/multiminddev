# Development Commands and Workflow

## Package Management (UV)
```bash
# Environment setup
uv venv
uv sync

# Package management (NEVER edit pyproject.toml directly)
uv add <package>
uv add --dev <dev-package>
uv remove <package>

# Running commands
uv run python script.py
uv run pytest
uv run ruff check .
```

## Quality Assurance Commands
```bash
# Format and lint
uv run ruff format .
uv run ruff check --fix .

# Type checking
uv run mypy src/

# Testing
uv run pytest
uv run pytest --cov=src/coding_framework --cov-report=html
uv run pytest -m "unit"           # Unit tests only
uv run pytest -m "integration"    # Integration tests only
uv run pytest -m "not slow"       # Skip slow tests
```

## Application Commands
```bash
# Main CLI usage
python -m coding_framework.cli solve "problem description" --language python
python -m coding_framework.cli review path/to/code.py
python -m coding_framework.cli execute path/to/script.py --timeout 30
python -m coding_framework.cli health
python -m coding_framework.cli train --algorithm ppo --episodes 100
```

## System Requirements
- Python 3.8+
- Docker (optional, for secure execution)
- Virtual environment: venv_linux (mentioned in CLAUDE.md)

## Code Quality Standards
- Line length: 100 characters (Ruff rule)
- Functions: <50 lines
- Classes: <100 lines  
- Files: <500 lines
- Always use type hints
- Google-style docstrings for public functions