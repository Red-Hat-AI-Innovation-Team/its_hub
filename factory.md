# Factory Configuration

## Goal

Evolve its_hub, a Python library for inference-time scaling of LLMs, improving test coverage, type safety, code quality, and capability surface across algorithms, reward models, and the IaaS service layer.

## Scope

### Modifiable

- its_hub/**/*.py
- tests/**/*.py
- eval/**/*.py

### Read-only

- README.md
- pyproject.toml
- CLAUDE.md
- docs/**/*.md

## Guards

- Do not delete or overwrite existing tests
- Do not modify files outside the declared scope
- Do not introduce secrets or credentials into the repository

## Eval

### Command

```bash
uv run python eval/score.py
```

### Threshold

0.50

## Target Branch

v1

## Smoke Test

```bash
uv run python -c "from its_hub import SelfConsistency, BestOfN; print('OK')"
```

## Constraints

- Prefer small, incremental changes over large rewrites
- Each change should be accompanied by at least one test
- Follow the existing code style and conventions
- Use uv for all Python tooling
- Always sign off commits with -s
