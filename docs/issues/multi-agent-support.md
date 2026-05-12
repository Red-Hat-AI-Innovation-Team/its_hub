# Issue: Add coding agent support for Gemini CLI, Codex, and OpenCode

## Status: RESOLVED

## Context

The its-hub plugin initially supported Claude Code and Cursor via `.claude-plugin/` and `.cursor-plugin/` manifest directories. Support for three additional coding agents was added using the superpowers plugin (v5.0.5) as a template.

## Implemented

- **Gemini CLI** — `gemini-extension.json` + `GEMINI.md` context file
- **Codex CLI** — `.codex/INSTALL.md` with symlink-based skill discovery
- **OpenCode** — `.opencode/plugins/its-hub.js` plugin module + `.opencode/INSTALL.md`

## Supported Agents

| Agent | Discovery Mechanism | Install Method |
|---|---|---|
| Claude Code | `.claude-plugin/plugin.json` | `/plugin install` or marketplace |
| Cursor | `.cursor-plugin/plugin.json` | Plugin marketplace |
| Gemini CLI | `gemini-extension.json` | `gemini extensions install <url>` |
| Codex CLI | Symlink to `~/.agents/skills/` | Clone + symlink |
| OpenCode | `.opencode/plugins/its-hub.js` | Add to `opencode.json` plugins array |

## Notes

- Core content (commands/, skills/, scripts/) is shared across all agents — only discovery/adapter files differ per platform.
- Commands (slash commands) are Claude Code/Cursor specific. Other agents access the same functionality via skills, which invoke the scripts directly.
- Skills reference `${CLAUDE_PLUGIN_ROOT}` for script paths. Each platform resolves this differently: Claude Code/Cursor expand it natively, Gemini CLI maps it via GEMINI.md to `${extensionPath}`, Codex resolves via clone path, OpenCode injects absolute paths via plugin JS.
