---
name: setup-guide
description: "Use when the user wants to set up inference-time scaling for the first time, or when its_hub is not yet installed/configured in the current environment."
---

# its_hub Setup Guide

You are helping the user set up inference-time scaling for the first time.

## Detection

First, detect the environment by running:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"
```

## If Nothing is Installed

1. Explain what its_hub does: "its_hub is a library for inference-time scaling — it generates multiple LLM responses and selects the best one using voting, scoring, or search algorithms."
2. Ask permission: "I can install it for you. This will add the `its_hub` Python package to your environment. Want me to proceed?"
3. If yes: install using the detected installer (`uv pip install its_hub` or `pip install its_hub`)
4. Proceed to configuration

## Configuration

Invoke the `/its-setup` command to walk through configuration:
- Provider, endpoint, API key, model
- Algorithm choice with explanations
- Algorithm-specific settings

## After Setup

Once configured, hand off to the `inference-scaling` skill if the user had an original scaling request. Otherwise, tell the user:
- "You're all set! You can now use `/its-scale <prompt>` to run inference-time scaling."
- Mention `/its-server` for server management and `/its-scale-batch` for batch processing.
