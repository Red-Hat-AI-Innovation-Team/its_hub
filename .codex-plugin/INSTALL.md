# Installing its-hub for Codex

Enable its-hub inference-time scaling skills in Codex via native skill discovery.

## Prerequisites

- Git
- Python 3.11+ with uv or pip
- [Codex CLI](https://github.com/openai/codex) installed

## Installation

1. **Clone the its-hub repository:**
   ```bash
   git clone https://github.com/redhat-ai-innovation/its_hub.git ~/.codex/its-hub
   ```

2. **Install the Python library:**
   ```bash
   cd ~/.codex/its-hub && uv sync --extra lm
   ```
   Or with pip:
   ```bash
   pip install "its_hub[lm]"
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
