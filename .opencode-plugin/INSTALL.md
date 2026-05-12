# Installing its-hub for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Python 3.10+ with uv or pip

## Installation

1. Add its-hub to the `plugin` array in your `opencode.json` (global or project-level):

   ```json
   {
     "plugin": ["its-hub@git+https://github.com/redhat-ai-innovation/its_hub.git"]
   }
   ```

2. Restart OpenCode. The plugin auto-installs and registers all skills.

3. Install the Python library (needed for scaling):
   ```bash
   uv pip install its_hub
   ```

Verify by asking: "What inference-time scaling skills do you have?"

## Usage

Use OpenCode's native `skill` tool:

```
use skill tool to list skills
use skill tool to load its-hub/inference-scaling
```

Or just describe what you want: "Scale this prompt using self-consistency with budget 8."

## Updating

Restart OpenCode to pull the latest version.

To pin a specific version:

```json
{
  "plugin": ["its-hub@git+https://github.com/redhat-ai-innovation/its_hub.git#v1.0.0"]
}
```

## Tool Mapping

When skills reference Claude Code tools:
- `Bash(...)` → your native shell execution tool
- `Read`, `Write`, `Edit` → your native file tools
- `/its-setup`, `/its-scale` commands → invoke the matching skill directly

## Troubleshooting

### Plugin not loading

1. Check logs: `opencode run --print-logs "hello" 2>&1 | grep -i its-hub`
2. Verify the plugin line in your `opencode.json`

### Skills not found

1. Use `skill` tool to list what's discovered
2. Check that the plugin is loading (see above)

## Uninstalling

Remove the `its-hub` entry from the `plugin` array in `opencode.json` and restart OpenCode.
