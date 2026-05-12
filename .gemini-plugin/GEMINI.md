@./skills/inference-scaling/SKILL.md
@./skills/setup-guide/SKILL.md

## Path Mapping

When skills reference `${CLAUDE_PLUGIN_ROOT}`, substitute `${extensionPath}` — this resolves to the its-hub extension directory where scripts are located.

Example: `"${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh"` becomes `"${extensionPath}/scripts/its_detect.sh"`

## Tool Mapping

When skills reference Claude Code concepts:
- `/its-setup`, `/its-scale`, `/its-server`, `/its-scale-batch` — invoke the matching skill or run the corresponding script in `scripts/` directly
- `Bash(...)` tool → your native shell execution tool
- `Read`, `Write`, `Edit` → your native file tools
