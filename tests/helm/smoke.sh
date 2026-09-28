#!/usr/bin/env bash
# Uses the selected disposable cluster. Never point this at a shared namespace.
set -euo pipefail
chart_dir="$(cd "$(dirname "$0")/../.." && pwd)/deploy/helm/its-hub"
namespace="${ITS_TEST_NAMESPACE:-its-helm-smoke}"
image="${ITS_TEST_IMAGE:-its-hub:helm-ci}"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
kubectl create namespace "$namespace"
kubectl -n "$namespace" create configmap mock-llm --from-file=server.py="$(dirname "$0")/mock_llm.py"
kubectl -n "$namespace" create secret generic upstream --from-literal=api-key=smoke-key
cat > "$work_dir/mock.yaml" <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mock-llm
spec:
  replicas: 1
  selector:
    matchLabels: {app: mock-llm}
  template:
    metadata:
      labels: {app: mock-llm}
    spec:
      containers:
        - name: mock
          image: $image
          imagePullPolicy: Never
          command: [python, /mock/server.py]
          ports: [{containerPort: 8000}]
          volumeMounts: [{name: script, mountPath: /mock}]
      volumes:
        - name: script
          configMap: {name: mock-llm}
---
apiVersion: v1
kind: Service
metadata:
  name: mock-llm
spec:
  selector: {app: mock-llm}
  ports: [{port: 8000, targetPort: 8000}]
YAML
kubectl -n "$namespace" apply -f "$work_dir/mock.yaml"
kubectl -n "$namespace" rollout status deployment/mock-llm --timeout=120s
cat > "$work_dir/values.yaml" <<YAML
image:
  repository: ${image%:*}
  tag: ${image##*:}
  pullPolicy: Never
config:
  enabled: true
  settings:
    endpoint: http://mock-llm:8000/v1
    model: smoke-model
    alg: self-consistency
    budget: 2
  apiKeySecret:
    name: upstream
    key: api-key
YAML
helm install smoke "$chart_dir" -n "$namespace" -f "$work_dir/values.yaml" --wait --timeout 3m
check_pods() {
  local pod_count=0
  for pod in $(kubectl -n "$namespace" get pods -l app.kubernetes.io/instance=smoke -o go-template='{{range .items}}{{if not .metadata.deletionTimestamp}}{{.metadata.name}}{{"\n"}}{{end}}{{end}}'); do
    pod_count=$((pod_count + 1))
    kubectl -n "$namespace" exec -i "$pod" -- python - <<'PY'
import json
import urllib.request
base = "http://localhost:8109"
for path in ("/health", "/ready"):
    assert urllib.request.urlopen(base + path).status == 200
models = json.load(urllib.request.urlopen(base + "/v1/models"))
assert models["data"][0]["id"] == "smoke-model", models
request = urllib.request.Request(base + "/v1/chat/completions", headers={"Content-Type": "application/json"}, data=json.dumps({"model": "smoke-model", "messages": [{"role": "user", "content": "What is 6 times 7?"}]}).encode())
response = json.load(urllib.request.urlopen(request, timeout=30))
assert response["choices"][0]["message"]["content"] == "42", response
print("Health, readiness, mounted credential and inference passed")
PY
  done
  test "$pod_count" -eq "$1"
}
check_pods 1
# Replacing a pod must restore defaults and the credential before readiness.
kubectl -n "$namespace" rollout restart deployment/smoke-its-hub
kubectl -n "$namespace" rollout status deployment/smoke-its-hub --timeout=180s
check_pods 1
# A non-image UID approximates OpenShift's arbitrary-UID runtime contract.
helm upgrade smoke "$chart_dir" -n "$namespace" -f "$work_dir/values.yaml" \
  --set replicaCount=2 --set config.settings.budget=3 --set securityContext.runAsUser=12345 \
  --wait --timeout 3m
check_pods 2
helm rollback smoke 1 -n "$namespace" --wait --timeout 3m
check_pods 1
helm uninstall smoke -n "$namespace" --wait --timeout 3m
if kubectl -n "$namespace" get deployment smoke-its-hub >/dev/null 2>&1; then
  echo 'Deployment remained after uninstall' >&2
  exit 1
fi
kubectl delete namespace "$namespace" --wait=false
