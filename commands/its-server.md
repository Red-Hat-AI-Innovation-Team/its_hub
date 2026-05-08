---
description: "Manage the IaaS inference-time scaling server"
argument-hint: "start|stop|status"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/its_detect.sh:*)"]
---

# its-hub Server Management

Manage the its_hub Inference-as-a-Service (IaaS) server.

## Usage

Run the server management script with the requested action:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/its_server.sh" $ARGUMENTS
```

Report the result to the user:
- **start**: Confirm the server is running, show the port and configured model/algorithm
- **stop**: Confirm the server has been stopped
- **status**: Show whether the server is running, its PID, port, and configured models

If no arguments are provided, default to `status`.

If the config is missing, suggest running `/its-setup` first.
