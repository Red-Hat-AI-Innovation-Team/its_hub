# Multi-Agent Plugin Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add plugin discovery and installation files for Gemini CLI, Codex CLI, and OpenCode — so its-hub can be installed as a plugin in all five major coding agents (Claude Code, Cursor, Gemini CLI, Codex, OpenCode).

**Architecture:** Each agent has its own discovery mechanism (manifest files, INSTALL docs, JS plugins), but the core content (skills/, commands/, scripts/) is shared and unchanged. The key portability challenge is that skills reference `${CLAUDE_PLUGIN_ROOT}` to locate scripts — each agent needs to resolve this to its own root path. Gemini CLI uses `${extensionPath}`, Codex uses symlinks, and OpenCode injects the path via a JS plugin. We add thin adapter files per platform following the superpowers plugin's proven patterns.

**Tech Stack:** JSON (manifests), Markdown (GEMINI.md, INSTALL docs), JavaScript/ESM (OpenCode plugin)

**Spec:** `docs/issues/multi-agent-support.md`

**Template:** Superpowers plugin at `/home/lab/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.5`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `gemini-extension.json` | Gemini CLI extension manifest (root level) |
| `GEMINI.md` | Gemini CLI context file — bootstraps plugin and maps `${CLAUDE_PLUGIN_ROOT}` to `${extensionPath}` |
| `.codex/INSTALL.md` | Codex CLI installation guide (clone + symlink) |
| `.opencode/INSTALL.md` | OpenCode installation guide (config-based) |
| `.opencode/plugins/its-hub.js` | OpenCode plugin module — registers skills dir and injects bootstrap context |

### Modified files

| File | Change |
|---|---|
| `README.md` | Add installation sections for Gemini CLI, Codex CLI, and OpenCode |
| `docs/issues/multi-agent-support.md` | Mark as resolved |

### Unchanged files (shared core)

| File | Why unchanged |
|---|---|
| `skills/inference-scaling/SKILL.md` | Uses `${CLAUDE_PLUGIN_ROOT}` — Claude Code/Cursor expand it natively; Gemini expands `${extensionPath}` via GEMINI.md mapping; Codex/OpenCode resolve via bootstrap context |
| `skills/setup-guide/SKILL.md` | Same as above |
| `commands/*.md` | Claude Code/Cursor only — other agents access functionality via skills |
| `scripts/*.sh` | Agent-agnostic bash — self-locating via `_env.sh` |

---

## Task 1: Gemini CLI Extension

**Files:**
- Create: `gemini-extension.json`
- Create: `GEMINI.md`

Template: `/home/lab/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.5/gemini-extension.json` and `GEMINI.md`

- [ ] **Step 1: Create the Gemini CLI manifest**

Create `gemini-extension.json` at the repo root:

```json
{
  "name": "its-hub",
  "description": "Inference-time scaling for LLMs — generate multiple candidates and select the best using voting, scoring, or search",
  "version": "1.0.0",
  "contextFileName": "GEMINI.md"
}
```

The `contextFileName` field tells Gemini CLI to load `GEMINI.md` as context injected into every conversation. Gemini CLI auto-discovers `skills/` directories containing `SKILL.md` files, so our existing skills will be found automatically.

- [ ] **Step 2: Create GEMINI.md context file**

Create `GEMINI.md` at the repo root. Following the superpowers pattern, use `@./` include directives to load skill content directly — this keeps the context file in sync with actual skills automatically, avoiding content drift.

```markdown
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
```

- [ ] **Step 3: Verify extension structure**

Run: `ls gemini-extension.json GEMINI.md`
Expected: both files exist

- [ ] **Step 4: Commit**

```bash
git add gemini-extension.json GEMINI.md
git commit -s -m "Add Gemini CLI extension manifest and context file"
```

---

## Task 2: Codex CLI Plugin

**Files:**
- Create: `.codex/INSTALL.md`

Template: `/home/lab/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.5/.codex/INSTALL.md`

Codex discovers skills by scanning `~/.agents/skills/`. Installation is clone + symlink — no manifest file needed beyond the INSTALL guide.

- [ ] **Step 1: Create the Codex installation guide**

```bash
mkdir -p .codex
```

Create `.codex/INSTALL.md`:

```markdown
# Installing its-hub for Codex

Enable its-hub inference-time scaling skills in Codex via native skill discovery.

## Prerequisites

- Git
- Python 3.10+ with uv or pip
- [Codex CLI](https://github.com/openai/codex) installed

## Installation

1. **Clone the its-hub repository:**
   ```bash
   git clone https://github.com/redhat-ai-innovation/its_hub.git ~/.codex/its-hub
   ```

2. **Install the Python library:**
   ```bash
   cd ~/.codex/its-hub && uv sync --extra dev
   ```
   Or with pip:
   ```bash
   pip install -e ~/.codex/its-hub
   ```

3. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/its-hub/skills ~/.agents/skills/its-hub
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\its-hub" "$env:USERPROFILE\.codex\its-hub\skills"
   ```

4. **Restart Codex** to discover the skills.

## Path Resolution

When skills reference `${CLAUDE_PLUGIN_ROOT}/scripts/...`, use the clone path instead:
```bash
~/.codex/its-hub/scripts/its_detect.sh
~/.codex/its-hub/scripts/its_server.sh
~/.codex/its-hub/scripts/its_scale.sh
```

## Verify

```bash
ls -la ~/.agents/skills/its-hub
```

You should see a symlink pointing to the its-hub skills directory.

## Updating

```bash
cd ~/.codex/its-hub && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/its-hub
rm -rf ~/.codex/its-hub
```
```

- [ ] **Step 2: Commit**

```bash
git add .codex/INSTALL.md
git commit -s -m "Add Codex CLI installation guide"
```

---

## Task 3: OpenCode Plugin

**Files:**
- Create: `.opencode/INSTALL.md`
- Create: `.opencode/plugins/its-hub.js`

Template: `/home/lab/.claude/plugins/cache/claude-plugins-official/superpowers/5.0.5/.opencode/INSTALL.md` and `.opencode/plugins/superpowers.js`

OpenCode uses a JS plugin module that registers the skills directory and injects bootstrap context into the system prompt. No symlinks needed — the plugin auto-installs when added to `opencode.json`.

- [ ] **Step 1: Create the OpenCode plugin module**

```bash
mkdir -p .opencode/plugins
```

Create `.opencode/plugins/its-hub.js`:

```javascript
/**
 * its-hub plugin for OpenCode.ai
 *
 * Registers skills directory and injects bootstrap context.
 * Tells the agent where to find its-hub scripts.
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const ItsHubPlugin = async ({ client, directory }) => {
  const pluginRoot = path.resolve(__dirname, '../..');
  const skillsDir = path.join(pluginRoot, 'skills');
  const scriptsDir = path.join(pluginRoot, 'scripts');

  const getBootstrapContent = () => {
    return `<its-hub-plugin>
You have the its-hub inference-time scaling plugin installed.

**Available skills:**
- inference-scaling — improve LLM response quality via multiple candidates
- setup-guide — first-time configuration

**Script paths (use these instead of \${CLAUDE_PLUGIN_ROOT}):**
- Detection: ${scriptsDir}/its_detect.sh
- Scaling: ${scriptsDir}/its_scale.sh
- Server: ${scriptsDir}/its_server.sh

When skills reference \${CLAUDE_PLUGIN_ROOT}/scripts/..., substitute the paths above.
</its-hub-plugin>`;
  };

  return {
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      if (!config.skills.paths.includes(skillsDir)) {
        config.skills.paths.push(skillsDir);
      }
    },

    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
```

- [ ] **Step 2: Create the OpenCode installation guide**

Create `.opencode/INSTALL.md`:

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add .opencode/INSTALL.md .opencode/plugins/its-hub.js
git commit -s -m "Add OpenCode plugin module and installation guide"
```

---

## Task 4: Update README with Multi-Agent Installation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

Read `README.md` to find the existing plugin installation section.

- [ ] **Step 2: Add installation sections for new agents**

After the existing Claude Code and Cursor installation sections, add:

```markdown
### Gemini CLI

Install as a Gemini CLI extension:

```bash
gemini extensions install https://github.com/redhat-ai-innovation/its_hub
```

This registers the extension and makes its-hub skills available in Gemini CLI conversations. See `GEMINI.md` for details.

### Codex CLI

Install via native skill discovery:

```bash
git clone https://github.com/redhat-ai-innovation/its_hub.git ~/.codex/its-hub
mkdir -p ~/.agents/skills
ln -s ~/.codex/its-hub/skills ~/.agents/skills/its-hub
```

Restart Codex to discover the skills. See `.codex/INSTALL.md` for full instructions.

### OpenCode

Add to your `opencode.json`:

```json
{
  "plugin": ["its-hub@git+https://github.com/redhat-ai-innovation/its_hub.git"]
}
```

Restart OpenCode. See `.opencode/INSTALL.md` for full instructions.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -s -m "Add Gemini CLI, Codex, and OpenCode installation docs to README"
```

---

## Task 5: Update Multi-Agent Issue and CLAUDE.md

**Files:**
- Modify: `docs/issues/multi-agent-support.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the tracking issue**

Replace the content of `docs/issues/multi-agent-support.md` with:

```markdown
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
```

- [ ] **Step 2: Update CLAUDE.md plugin section**

In `CLAUDE.md`, update the Plugin Structure section to mention all five supported agents. After the existing structure listing, add:

```markdown
### Multi-Agent Support

The plugin supports five coding agents. Core content is shared; only discovery files differ:

| Agent | Discovery File |
|---|---|
| Claude Code | `.claude-plugin/plugin.json` |
| Cursor | `.cursor-plugin/plugin.json` |
| Gemini CLI | `gemini-extension.json` + `GEMINI.md` |
| Codex CLI | `.codex/INSTALL.md` (symlink-based) |
| OpenCode | `.opencode/plugins/its-hub.js` |
```

- [ ] **Step 3: Commit**

```bash
git add docs/issues/multi-agent-support.md CLAUDE.md
git commit -s -m "Update multi-agent tracking issue and CLAUDE.md with new agent support"
```

---

## Task 6: Final Verification

- [ ] **Step 1: Verify all new files exist**

```bash
ls gemini-extension.json GEMINI.md .codex/INSTALL.md .opencode/INSTALL.md .opencode/plugins/its-hub.js
```

Expected: all five files listed.

- [ ] **Step 2: Validate JSON syntax**

```bash
python3 -c "import json; json.load(open('gemini-extension.json'))" && echo "OK"
```

- [ ] **Step 3: Verify JS syntax**

```bash
node --check .opencode/plugins/its-hub.js 2>&1 || echo "Syntax check not available — visual review OK"
```

- [ ] **Step 4: Verify existing tests still pass**

```bash
uv run pytest tests/ -q --tb=short
```

Expected: 230 passed, 1 skipped

- [ ] **Step 5: Verify lint is clean**

```bash
uv run ruff check its_hub/ && uv run ruff format --check its_hub/
```

- [ ] **Step 6: Review full plugin file set**

```bash
find .claude-plugin .cursor-plugin .codex .opencode gemini-extension.json GEMINI.md -type f 2>/dev/null | sort
```

Expected:
```
.claude-plugin/marketplace.json
.claude-plugin/plugin.json
.codex/INSTALL.md
.cursor-plugin/plugin.json
gemini-extension.json
GEMINI.md
.opencode/INSTALL.md
.opencode/plugins/its-hub.js
```

- [ ] **Step 7: Git status — verify clean**

```bash
git status
```
