# Issue: Add coding agent support for Gemini CLI, Codex, and OpenCode

## Context

The its-hub plugin currently supports Claude Code and Cursor via `.claude-plugin/` and `.cursor-plugin/` manifest directories. The core plugin content (skills, commands, scripts) is agent-agnostic markdown — only the discovery/manifest files differ per platform.

## Requested

Add manifest and adapter files for:

- **Gemini CLI** — `gemini-extension.json` + `GEMINI.md`
- **Codex** — `.codex/INSTALL.md`
- **OpenCode** — `.opencode/INSTALL.md`

## Reference

The `superpowers` plugin (v5.0.5) supports all five platforms and can be used as a template for the adapter file formats.

## Notes

- Core content (commands/, skills/, scripts/) should not change — only add platform-specific discovery files.
- Test each integration with the respective coding agent to verify skill/command discovery works.
