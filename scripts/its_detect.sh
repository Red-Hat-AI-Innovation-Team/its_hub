#!/usr/bin/env bash
# Detect its_hub environment: server, library, installer, config
set -euo pipefail

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"
IAAS_PORT=8108

# Read port from config if available
if [ -f "$CONFIG_PATH" ]; then
    CONFIGURED_PORT=$(CONFIG_PATH="$CONFIG_PATH" python3 -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('iaas_port', 8108))" 2>/dev/null || echo 8108)
    IAAS_PORT="$CONFIGURED_PORT"
fi

# Check IaaS server
if curl -s --connect-timeout 2 "http://localhost:${IAAS_PORT}/v1/models" > /dev/null 2>&1; then
    echo "server=running"
else
    echo "server=stopped"
fi

# Check library
if python3 -c "import its_hub" 2>/dev/null; then
    echo "library=installed"
else
    echo "library=missing"
fi

# Check installer
if command -v uv > /dev/null 2>&1; then
    echo "installer=uv"
elif command -v pip > /dev/null 2>&1; then
    echo "installer=pip"
else
    echo "installer=none"
fi

# Check config
if [ -f "$CONFIG_PATH" ]; then
    echo "config=found"
else
    echo "config=missing"
fi
