# Installing its-hub for Codex

## Via Marketplace (Recommended)

```bash
codex plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
```

Then install the plugin from the marketplace. The Python library will need to be installed separately:

```bash
pip install "its_hub[lm]"
```

## Manual Installation

If you prefer to install manually:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git ~/.codex/its-hub
   ```

2. **Install the Python library:**
   ```bash
   pip install "its_hub[lm]"
   ```

3. **Create the skills symlink** (skills are in `.claude/skills/` — shared between Claude Code and Codex):
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/its-hub/.claude/skills ~/.agents/skills/its-hub
   ```

4. **Restart Codex** to discover the skills.

## Updating

Marketplace installs update automatically. For manual installs:
```bash
cd ~/.codex/its-hub && git pull
```

## Uninstalling

For manual uninstalls:
```bash
rm ~/.agents/skills/its-hub
rm -rf ~/.codex/its-hub
```
