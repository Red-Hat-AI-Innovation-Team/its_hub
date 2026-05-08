#!/usr/bin/env bash
# Manage its_hub IaaS server lifecycle
set -euo pipefail

CONFIG_PATH="${ITS_HUB_CONFIG:-.its-hub/config.json}"
PID_FILE=".its-hub/server.pid"
ACTION="${1:-status}"

die() { echo "ERROR: $1" >&2; exit 1; }

read_config() {
    [ -f "$CONFIG_PATH" ] || die "Config not found at $CONFIG_PATH. Run /its-setup first."
    CONFIG_PATH="$CONFIG_PATH" FIELD="$1" DEFAULT="${2:-}" python3 -c "
import json, os
c = json.load(open(os.environ['CONFIG_PATH']))
print(c.get(os.environ['FIELD'], os.environ['DEFAULT']))
"
}

build_configure_payload() {
    CONFIG_PATH="$CONFIG_PATH" MODEL_KEY="${1:-default}" python3 -c "
import json, sys, os

config = json.load(open(os.environ['CONFIG_PATH']))
model_key = os.environ['MODEL_KEY']
model_cfg = config.get('models', {}).get(model_key, {})
alg_cfg = config.get('algorithm_config', {})

payload = {
    'provider': config.get('provider', 'openai'),
    'endpoint': model_cfg.get('endpoint', ''),
    'api_key': model_cfg.get('api_key'),
    'model': model_cfg.get('model', ''),
    'alg': config.get('algorithm', ''),
}

# Add extra_args if present
extra = config.get('extra_args')
if extra:
    payload['extra_args'] = extra

# Map algorithm_config fields to flat /configure fields
field_map = {
    'regex_patterns': 'regex_patterns',
    'tool_vote': 'tool_vote',
    'exclude_tool_args': 'exclude_tool_args',
    'rm_name': 'rm_name',
    'rm_device': 'rm_device',
    'rm_agg_method': 'rm_agg_method',
    'step_token': 'step_token',
    'stop_token': 'stop_token',
    'judge_model': 'judge_model',
    'judge_base_url': 'judge_base_url',
    'judge_api_key': 'judge_api_key',
    'judge_criterion': 'judge_criterion',
    'judge_mode': 'judge_mode',
    'judge_top_n': 'judge_top_n',
    'judge_temperature': 'judge_temperature',
    'judge_max_tokens': 'judge_max_tokens',
    'enable_judge_logging': 'enable_judge_logging',
}

for src, dst in field_map.items():
    val = alg_cfg.get(src)
    if val is not None:
        payload[dst] = val

# Remove None values
payload = {k: v for k, v in payload.items() if v is not None}

json.dump(payload, sys.stdout)
"
}

case "$ACTION" in
    start)
        # Get port from config if available, otherwise use default
        if [ -f "$CONFIG_PATH" ]; then
            IAAS_PORT=$(read_config iaas_port 8108)
        else
            IAAS_PORT=8108
        fi

        # Check if already running
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Server already running (PID $(cat "$PID_FILE")) on port $IAAS_PORT"
            exit 0
        fi

        mkdir -p .its-hub

        echo "Starting IaaS server on port $IAAS_PORT..."
        nohup python3 -m its_hub.integration.iaas --host 0.0.0.0 --port "$IAAS_PORT" \
            > .its-hub/server.log 2>&1 &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PID_FILE"

        # Wait for server to be ready
        for i in $(seq 1 30); do
            if curl -s --connect-timeout 1 "http://localhost:${IAAS_PORT}/v1/models" > /dev/null 2>&1; then
                echo "Server started (PID $SERVER_PID)"

                # Configure only if config exists
                if [ -f "$CONFIG_PATH" ]; then
                    PAYLOAD=$(build_configure_payload)
                    RESPONSE=$(curl -s -X POST "http://localhost:${IAAS_PORT}/configure" \
                        -H "Content-Type: application/json" \
                        -d "$PAYLOAD")

                    if echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='success'" 2>/dev/null; then
                        echo "Server configured: $(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('message',''))")"
                    else
                        echo "WARNING: Server started but configuration failed: $RESPONSE"
                    fi
                else
                    echo "Server started (unconfigured). Run /its-setup to configure."
                fi
                exit 0
            fi
            sleep 1
        done

        echo "ERROR: Server failed to start within 30 seconds. Check .its-hub/server.log"
        kill "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        exit 1
        ;;

    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "No server PID file found. Server may not be running."
            exit 0
        fi

        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Server stopped (PID $PID)"
        else
            echo "Server process $PID not found (stale PID file). Cleaning up."
        fi
        rm -f "$PID_FILE"
        ;;

    status)
        IAAS_PORT=$(read_config iaas_port 8108 2>/dev/null || echo 8108)

        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            PID=$(cat "$PID_FILE")
            MODELS=$(curl -s --connect-timeout 2 "http://localhost:${IAAS_PORT}/v1/models" 2>/dev/null || echo '{"data":[]}')
            echo "server=running pid=$PID port=$IAAS_PORT"
            echo "models=$MODELS"
        else
            echo "server=stopped"
            [ -f "$PID_FILE" ] && echo "(stale PID file found — cleaning up)" && rm -f "$PID_FILE"
        fi
        ;;

    *)
        echo "Usage: its_server.sh {start|stop|status}"
        exit 1
        ;;
esac
