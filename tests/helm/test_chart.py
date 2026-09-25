"""Render checks runnable without importing ITS or requiring a cluster."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

CHART = Path(__file__).resolve().parents[2] / "deploy/helm/its-hub"
CONFIG = {
    "enabled": True,
    "settings": {
        "endpoint": "http://llm:8000/v1",
        "model": "test",
        "alg": "self-consistency",
    },
}


def render(values=None, *, valid=True, openshift=False):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        values = dict(values or {})
        values["image"] = {"tag": "ci", **values.get("image", {})}
        yaml.safe_dump(values, f)
        f.flush()
        command = [
            "helm",
            "template",
            "test",
            str(CHART),
            "-f",
            f.name,
        ]
        if openshift:
            command += ["--api-versions", "route.openshift.io/v1"]
        result = subprocess.run(command, capture_output=True, text=True)
    if not valid:
        assert result.returncode != 0, result.stdout
        return result.stderr
    assert result.returncode == 0, result.stderr
    return {d["kind"]: d for d in yaml.safe_load_all(result.stdout) if d}


@unittest.skipUnless(
    shutil.which("helm"), "Helm CLI is required for chart render tests"
)
class ChartTests(unittest.TestCase):
    def test_defaults(self):
        docs = render()
        pod = docs["Deployment"]["spec"]["template"]["spec"]
        container = pod["containers"][0]
        self.assertEqual(container["readinessProbe"]["httpGet"]["path"], "/ready")
        self.assertEqual(container["livenessProbe"]["httpGet"]["path"], "/health")
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(
            docs["Service"]["spec"]["selector"],
            docs["Deployment"]["spec"]["selector"]["matchLabels"],
        )
        # Deployment keeps the bare fullname; others carry a kind suffix.
        self.assertEqual(docs["Deployment"]["metadata"]["name"], "test-its-hub")
        self.assertEqual(docs["Service"]["metadata"]["name"], "test-its-hub-svc")
        self.assertEqual(docs["ServiceAccount"]["metadata"]["name"], "test-its-hub-sa")
        self.assertEqual(pod["serviceAccountName"], "test-its-hub-sa")
        self.assertNotIn("ConfigMap", docs)
        self.assertNotIn("Secret", docs)

    def test_config_and_secret_reference(self):
        docs = render(
            {
                "config": {
                    **CONFIG,
                    "apiKeySecret": {"name": "provider", "key": "token"},
                },
                "replicaCount": 2,
            }
        )
        self.assertEqual(docs["ConfigMap"]["metadata"]["name"], "test-its-hub-config")
        self.assertEqual(
            json.loads(docs["ConfigMap"]["data"]["config.json"]), CONFIG["settings"]
        )
        pod = docs["Deployment"]["spec"]["template"]["spec"]
        config_vol = next(v["configMap"] for v in pod["volumes"] if "configMap" in v)
        self.assertEqual(config_vol["name"], "test-its-hub-config")
        secret = next(v["secret"] for v in pod["volumes"] if "secret" in v)
        self.assertEqual(secret["secretName"], "provider")
        self.assertEqual(secret["items"], [{"key": "token", "path": "api-key"}])
        self.assertNotIn("Secret", docs)

    def test_rollout_checksum(self):
        first = render({"config": CONFIG})
        second = render(
            {"config": {**CONFIG, "settings": {**CONFIG["settings"], "budget": 8}}}
        )

        def checksum(docs):
            return docs["Deployment"]["spec"]["template"]["metadata"]["annotations"][
                "checksum/config"
            ]

        self.assertNotEqual(checksum(first), checksum(second))

    def test_digest_precedence(self):
        digest = "sha256:" + "a" * 64
        pod = render({"image": {"digest": digest}})["Deployment"]["spec"]["template"][
            "spec"
        ]
        self.assertTrue(pod["containers"][0]["image"].endswith("@" + digest))

    def test_optional_resources_and_private_admin(self):
        # Off OpenShift, expose auto-detects Ingress.
        docs = render(
            {
                "config": CONFIG,
                "autoscaling": {"enabled": True},
                "podDisruptionBudget": {"enabled": True},
                "expose": {"enabled": True, "host": "its.example.com"},
                "networkPolicy": {"enabled": True, "restrictEgress": True},
            }
        )
        self.assertNotIn("replicas", docs["Deployment"]["spec"])
        self.assertIn("HorizontalPodAutoscaler", docs)
        self.assertIn("PodDisruptionBudget", docs)
        hpa = docs["HorizontalPodAutoscaler"]
        self.assertEqual(hpa["metadata"]["name"], "test-its-hub-hpa")
        # HPA must target the Deployment by its bare name.
        self.assertEqual(hpa["spec"]["scaleTargetRef"]["name"], "test-its-hub")
        self.assertEqual(docs["Ingress"]["metadata"]["name"], "test-its-hub-ingress")
        self.assertEqual(
            docs["Ingress"]["spec"]["rules"][0]["http"]["paths"][0]["backend"][
                "service"
            ]["name"],
            "test-its-hub-svc",
        )
        self.assertNotIn("Route", docs)
        self.assertEqual(
            docs["Ingress"]["spec"]["rules"][0]["http"]["paths"][0]["path"], "/v1"
        )
        self.assertEqual(docs["NetworkPolicy"]["spec"]["ingress"], [])
        self.assertEqual(docs["NetworkPolicy"]["spec"]["egress"], [])

    def test_openshift_autodetects_route_and_assigned_uid(self):
        # On OpenShift, expose auto-detects Route; no host required.
        docs = render(
            {"expose": {"enabled": True}, "securityContext": {"runAsUser": None}},
            openshift=True,
        )
        self.assertNotIn(
            "runAsUser",
            docs["Deployment"]["spec"]["template"]["spec"]["containers"][0][
                "securityContext"
            ],
        )
        self.assertNotIn("Ingress", docs)
        self.assertEqual(docs["Route"]["metadata"]["name"], "test-its-hub-route")
        self.assertEqual(docs["Route"]["spec"]["to"]["name"], "test-its-hub-svc")
        self.assertEqual(docs["Route"]["spec"]["path"], "/v1/")

    def test_expose_kind_forces_ingress_on_openshift(self):
        docs = render(
            {"expose": {"enabled": True, "kind": "ingress", "host": "its.example.com"}},
            openshift=True,
        )
        self.assertNotIn("Route", docs)
        self.assertIn("Ingress", docs)

    def test_invalid_values(self):
        cases = [
            {"image": {"tag": ""}},
            {"replicaCount": 2},
            {"config": {"enabled": True}},
            {
                "config": {
                    **CONFIG,
                    "settings": {**CONFIG["settings"], "api_key": "do-not-store"},
                }
            },
            {"config": {"apiKeySecret": {"name": "provider"}}},
            {"config": {**CONFIG, "settings": {**CONFIG["settings"], "budget": 0}}},
            {"expose": {"enabled": True}},  # auto -> Ingress off OCP, host required
            {"expose": {"enabled": True, "kind": "route"}},  # no route API available
            {"expose": {"kind": "bogus"}},  # not in schema enum
            {"autoscaling": {"minReplicas": 5, "maxReplicas": 2}},
        ]
        for values in cases:
            with self.subTest(values=values):
                render(values, valid=False)


if __name__ == "__main__":
    unittest.main()
