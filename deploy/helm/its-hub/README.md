# ITS-Hub Helm Chart

Deploys the ITS microservice, with an external OpenAI-compatible LLM endpoint.
Requires Helm 3+ and Kubernetes 1.26+.

## Install

Publish the service image using the repository's container Publish workflow,
or build and push it to your own registry. Set a published image tag or SHA256 digest
explicitly; the chart intentionally has no `latest` or unverified version default.

Edit `examples/configured.yaml` with your upstream endpoint and model, then run:

```bash
helm upgrade --install its deploy/helm/its-hub \
  --namespace its --create-namespace \
  --set-string image.tag=YOUR_PUBLISHED_TAG \
  -f deploy/helm/its-hub/examples/configured.yaml --wait --timeout 5m
kubectl -n its port-forward service/its-its-hub 8109:8109
curl http://localhost:8109/ready
curl http://localhost:8109/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"What is 6 times 7?"}]}'
```

For immutable deployments set `image.digest=sha256:...`; digest takes precedence over
tag. Configure `imagePullSecrets` for private registries. With configuration disabled
(the default), clients must supply `X-ITS-Endpoint` per request or configure the pod
manually. Readiness in that mode means the application is initialized, not that a
service-wide endpoint/model has been supplied.

The example above reaches the ClusterIP service via `port-forward`. To publish the
`/v1` API externally, set `expose.enabled=true`. With the default `expose.kind=auto`
the chart renders an OpenShift `Route` (no host needed) or, on plain Kubernetes, an
`Ingress` (host required):

```bash
# OpenShift: auto-detects a Route
helm upgrade --install its deploy/helm/its-hub -n its --create-namespace \
  --set-string image.tag=YOUR_PUBLISHED_TAG \
  -f deploy/helm/its-hub/examples/configured.yaml \
  -f deploy/helm/its-hub/examples/openshift.yaml --wait --timeout 5m

# Plain Kubernetes: auto-detects an Ingress (host is required)
helm upgrade --install its deploy/helm/its-hub -n its --create-namespace \
  --set-string image.tag=YOUR_PUBLISHED_TAG \
  -f deploy/helm/its-hub/examples/configured.yaml \
  --set expose.enabled=true --set expose.host=its.example.com --wait --timeout 5m
```

These resources add no client authentication; put TLS and auth at your gateway before
exposing the API to untrusted callers.

## Credentials

The chart references an **existing Secret in the release namespace**; it never
renders or stores API keys in Helm values or ConfigMaps. Create the Secret through
your secret manager, or from a local credential file:

```bash
kubectl create namespace its
kubectl -n its create secret generic its-upstream --from-file=api-key=/secure/path/provider-key
```

Use `examples/authenticated.yaml` with your model and published image tag. The selected
Secret key is mounted read-only and read once at startup. Without a Secret reference,
an unauthenticated upstream or per-request `X-ITS-API-Key` remains supported. This
credential authenticates ITS to the upstream.

## Configuration and readiness

`config.settings` uses the `/configure` JSON fields: `endpoint`, `model`, `alg`, and
optional `budget`, `temperature`, `regex_patterns`, `tool_vote`, `exclude_tool_args`,
`threshold`, and `confidence_threshold`. `api_key` is rejected in chart values.

The service reads `ITS_IAAS_CONFIG_FILE` as JSON and optionally reads the credential
from `ITS_IAAS_API_KEY_FILE`. It validates and applies settings before serving traffic;
missing files, invalid configuration, and empty credential files fail startup without
printing their contents. Omitting both variables preserves manual configuration.

- `/health`: startup and liveness probe.
- `/ready`: returns 200 after successful initialization, 503 before initialization
  and during lifespan teardown. Uvicorn also stops accepting connections during shutdown.
- Neither endpoint calls the LLM provider; an upstream outage must not cause restart loops.

Every replacement or additional pod loads the same settings. ConfigMap changes trigger
a rolling update through a pod-template checksum. Helm rejects multiple replicas or
HPA without managed configuration. Avoid runtime `/configure` calls on managed
replicas: they affect just one process and are overwritten on restart. Use Helm values
as the source of truth.

Secret contents are external to Helm; after rotating the Secret, restart the pods:

```bash
kubectl -n its rollout restart deployment/its-its-hub
kubectl -n its rollout status deployment/its-its-hub
```

## Networking and OpenShift

The Service is ClusterIP. Set `expose.enabled=true` to publish only the `/v1` API paths
externally, keeping `/configure`, `/docs`, and probe endpoints off that public route.
With the default `expose.kind=auto`, the chart renders an OpenShift `Route` on clusters
that expose `route.openshift.io/v1` and an `Ingress` everywhere else; force one with
`expose.kind: route|ingress`. `expose.host` is required for Ingress and optional for
Route (OpenShift assigns a host when empty). Ingress-only settings live under
`expose.ingress` (`className`, `annotations`, `tls`); Route-only settings under
`expose.route` (`annotations`, `tls`). Configure TLS and client authentication at your
gateway before exposing the API to untrusted clients. Pod/Service access still includes
the unauthenticated `/configure` endpoint; restrict that access to trusted workloads.
NetworkPolicy works at the network layer and cannot filter individual HTTP paths.

For OpenShift, layer `examples/openshift.yaml` over your configured values. It enables
`expose` (auto-detected as a Route) and removes the fixed UID needed to validate the
image's named USER on standard Kubernetes, letting the namespace SCC assign an arbitrary
UID. The Route defaults to edge TLS and redirects HTTP to HTTPS. The chart does not
create SCCs or require elevated privileges.

Inference can take several minutes. The Route timeout defaults to 300 seconds; set
controller-specific Ingress annotations for comparable upstream timeouts and streaming
support. The pod termination grace period defaults to 330 seconds for in-flight requests.

NetworkPolicy is opt-in. Enabling it with no `ingressFrom` denies inbound workload
traffic. Supply allowed namespace/pod selectors (including the ingress controller when
used). Egress stays unrestricted unless `restrictEgress=true`; then explicitly allow
cluster DNS and the upstream LLM. Your CNI must enforce NetworkPolicy.

## Main values

| Value | Purpose |
| --- | --- |
| `image.repository`, `tag`, `digest`, `pullPolicy` | Service image selection |
| `config.enabled`, `settings`, `apiKeySecret` | Reproducible startup settings and credential reference |
| `replicaCount` | Replica count, default 1 |
| `resources` | CPU/memory requests and limits; tune for traffic and inference budget |
| `probes` | Startup, liveness, and readiness timing |
| `securityContext`, `podSecurityContext` | Nonroot runtime, read-only root filesystem, dropped capabilities |
| `serviceAccount` | Create or reuse a ServiceAccount; token mounting is disabled |
| `expose` | Optional external API access; auto-selects Route (OpenShift) or Ingress |
| `autoscaling` | CPU HPA; requires managed config, CPU requests and cluster metrics |
| `podDisruptionBudget` | Optional disruption budget; use multiple replicas for availability |
| `networkPolicy` | Optional ingress/egress restrictions |
| `nodeSelector`, `tolerations`, `affinity`, `topologySpreadConstraints` | Scheduling |
| `podAnnotations` | Additional annotations, including external Secret reloader integrations |

## Upgrade, rollback, and uninstall

```bash
helm upgrade its deploy/helm/its-hub -n its -f my-values.yaml --wait --timeout 5m
helm history its -n its
helm rollback its PREVIOUS_REVISION -n its --wait --timeout 5m
helm uninstall its -n its --wait
```

Keep the pinned image and full configuration in `my-values.yaml`. Rollback restores
chart-managed configuration, not externally managed Secret contents. Uninstall leaves
existing Secrets and the namespace intact.

If pods are not ready, inspect `kubectl describe pod` and logs. Check that the image
contains `/ready`, configuration fields are valid, the referenced Secret/key exists,
and the registry is accessible. If inference fails while readiness succeeds, check
upstream DNS/connectivity, credentials, model name, and request timeouts.

## Validation and publishing

```bash
helm lint deploy/helm/its-hub --strict --set image.tag=validation
python tests/helm/test_chart.py
helm package deploy/helm/its-hub --destination /tmp/charts
```

The Helm workflow validates rendered schemas and runs `tests/helm/smoke.sh` in a
throwaway Kind cluster with a locally built image and authenticated mock upstream.
It covers install, inference, pod replacement, two replicas, arbitrary UID, upgrade,
rollback, and uninstall. The arbitrary-UID test does not substitute for a full
OpenShift Route/SCC acceptance test on an actual OpenShift cluster.

Chart versioning is independent of image versioning. Bump `Chart.yaml` for each chart
release. After validation, manually dispatch the **Helm** workflow on `main` with
`publish=true` to publish to `oci://ghcr.io/red-hat-ai-innovation-team/charts/its-hub`.
Publishing is not automatic on pushes or pull requests. Once published:

```bash
helm install its oci://ghcr.io/red-hat-ai-innovation-team/charts/its-hub \
  --version 0.1.0 -n its --create-namespace -f my-values.yaml
```
