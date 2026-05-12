# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
uv sync --extra dev                # Development installation (recommended)
pip install -e ".[dev]"            # Alternative: pip installation
```

### Testing
```bash
uv run pytest tests/                              # Run all tests
uv run pytest tests/test_algorithms.py             # Run a specific test file
uv run pytest tests/test_algorithms.py -k "test_self_consistency"  # Run a single test by name
uv run pytest tests/ --cov=its_hub                 # With coverage
uv run pytest tests/ -v                            # Verbose output
```

### Code Quality
```bash
uv run ruff check its_hub/                         # Lint
uv run ruff check its_hub/ --fix                   # Lint with auto-fix
uv run ruff format its_hub/                        # Format
uv run ruff format --check its_hub/                # Check formatting without modifying
```

### IaaS Service
```bash
uv run its-iaas --host 0.0.0.0 --port 8108        # Start service
just iaas-start                                    # Same, via justfile
curl -s http://localhost:8108/v1/models | jq .      # Health check
```

### Git Workflow
```bash
git commit -s -m "commit message"   # Always use sign-off (-s)
```

## Contribution Rules

- Always use the sign-off flag (`-s`) on git commits.
- Never mention Claude Code in commits or PRs. No `Co-Authored-By` lines referencing Claude.
- Prefer `rg` over `grep` when available.
- Use `uv` for all Python operations: `uv sync` to init, `uv run` to execute.

## CI Expectations

CI runs on Python 3.10, 3.11, and 3.12. It checks:
- `ruff check its_hub/` (linting)
- `ruff format --check its_hub/` (formatting)
- `pytest tests/ --cov=its_hub` (tests with coverage)

Ruff config: line-length 88, target py311, double quotes, space indent. See `ruff.toml` for full rule set.

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs. It generates multiple candidate responses and uses scoring/voting to select the best one.

### Core Design

All components are built on abstract base classes in `its_hub/base.py`:
- `AbstractLanguageModel` — async-first: subclasses implement `agenerate()`, and `generate()` is a sync wrapper via `asyncio.run()`
- `AbstractScalingAlgorithm` — same pattern: implement `ainfer()`, get `infer()` for free
- `AbstractOutcomeRewardModel` / `AbstractProcessRewardModel` — score full responses or individual reasoning steps, respectively
- `AbstractScalingResult` — wraps algorithm output; `.the_one` returns the selected response

### Algorithms (`its_hub/algorithms/`)

All share the same interface: `ainfer(lm, prompt, budget, return_response_only=True, tools=None, tool_choice=None)`

| Algorithm | Strategy | Budget meaning | Reward model |
|---|---|---|---|
| Self-Consistency | Generate N, majority vote | N parallel generations | None (voting) |
| Best-of-N | Generate N, rank by score | N parallel generations | Outcome RM |
| Beam Search | Step-by-step with beam width | depth = budget / beam_width | Process RM |
| Particle Filtering/Gibbs | Probabilistic resampling | Number of particles | Process RM |

**Planning Wrapper** (`planning_wrapper.py`) wraps any algorithm to add multi-step planning.

### Language Models (`its_hub/lms.py`)

- `OpenAICompatibleLanguageModel` — primary implementation, works with vLLM and OpenAI-compatible APIs
- `LiteLLMLanguageModel` — multi-provider support (AWS Bedrock, Google Vertex, etc.)
- `StepGeneration` — incremental generation with configurable step tokens (e.g., `"\n\n"`) and stop conditions; used by beam search and particle filtering

### Integration (`its_hub/integration/`)

- `reward_hub.py` — wraps `reward_hub` library for process reward models
- `iaas.py` — FastAPI server providing an OpenAI-compatible chat completions API with an added `budget` parameter for inference-time scaling. Supports tool/function calling and configurable voting strategies (`tool_vote`: `tool_name`, `tool_hierarchical`)

### Type System (`its_hub/types.py`)

`ChatMessage` and `ChatMessages` handle both string prompts and structured conversation history. `ChatMessages.from_prompt_or_messages()` is the standard entry point for normalizing input across the codebase.

### Error Handling (`its_hub/error_handling.py`)

API errors are classified as retryable (`RateLimitError`, `APIConnectionError`, `InternalServerError`) or non-retryable (`ContextLengthError`, `AuthenticationError`, `BadRequestError`). The `backoff` library handles retry logic with `should_retry()` as the gating function.

## Coding Agent Plugin

This repo also serves as a plugin for five coding agents. The plugin files are at the repo root:

### Plugin Structure

```
commands/                     # Slash commands (markdown files)
├── its-setup.md              # /its-setup — guided first-run configuration
├── its-scale.md              # /its-scale — single prompt scaling
├── its-scale-batch.md        # /its-scale-batch — batch scaling from file
└── its-server.md             # /its-server — IaaS server lifecycle

skills/                       # Contextual skills (fire automatically)
├── inference-scaling/
│   └── SKILL.md              # Detects scaling intent, routes to commands
└── setup-guide/
    └── SKILL.md              # First-time setup walkthrough

scripts/                      # Shell scripts used by commands/skills
├── its_detect.sh             # Environment detection (server, library, config)
├── its_server.sh             # IaaS server start/stop/status
└── its_scale.sh              # Execute scaling requests

.claude-plugin/               # Claude Code manifest
.cursor-plugin/               # Cursor manifest
.gemini-plugin/               # Gemini CLI context (GEMINI.md)
.codex-plugin/                # Codex CLI manifest + install guide
.opencode-plugin/             # OpenCode JS plugin module + install guide
gemini-extension.json         # Gemini CLI manifest (must be at root)
```

### Plugin Config

User config is stored at `.its-hub/config.json` (auto-generated by `/its-setup`, gitignored). See `docs/superpowers/specs/2026-05-08-claude-code-plugin-design.md` for the full config schema and IaaS field mapping.

### Multi-Agent Support

The plugin supports five coding agents. Core content is shared; only discovery files differ:

| Agent | Discovery File |
|---|---|
| Claude Code | `.claude-plugin/plugin.json` |
| Cursor | `.cursor-plugin/plugin.json` |
| Gemini CLI | `gemini-extension.json` + `.gemini-plugin/GEMINI.md` |
| Codex CLI | `.codex-plugin/plugin.json` |
| OpenCode | `.opencode-plugin/plugins/its-hub.js` |

### Plugin Development

- Skills are markdown files — edit `skills/*/SKILL.md` directly
- Commands are markdown files — edit `commands/*.md` directly
- Scripts are bash — edit `scripts/*.sh`
- Test plugin changes by running commands in any supported coding agent
- Plugin manifests and adapter files rarely change

## Dependency Notes

If `reward_hub` or `vllm` cause dependency conflicts during development, comment them out in `pyproject.toml` temporarily and retry `uv sync --extra dev`.
