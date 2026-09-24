"""Startup configuration is applied on every process, before readiness."""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from its_hub.integration.iaas.app import _lifespan, _state, app, health, ready


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("ITS_IAAS_CONFIG_FILE", raising=False)
    monkeypatch.delenv("ITS_IAAS_API_KEY_FILE", raising=False)
    _state.reset()
    yield
    _state.reset()


def write_config(tmp_path, monkeypatch, **overrides):
    data = {
        "endpoint": "http://upstream:8000/v1",
        "model": "test-model",
        "alg": "self-consistency",
        "budget": 3,
        **overrides,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("ITS_IAAS_CONFIG_FILE", str(path))
    return path


@pytest.mark.asyncio
async def test_readiness_lifecycle():
    with pytest.raises(HTTPException) as exc:
        await ready()
    assert exc.value.status_code == 503
    async with _lifespan(app):
        assert await ready() == {"status": "ready"}
    with pytest.raises(HTTPException) as exc:
        await ready()
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_health_independent_of_readiness():
    # Liveness must report OK before initialization, when /ready is still 503.
    assert await health() == {"status": "ok"}
    with pytest.raises(HTTPException) as exc:
        await ready()
    assert exc.value.status_code == 503
    assert TestClient(app).get("/health").status_code == 200


def test_unconfigured_startup_preserves_manual_mode():
    assert TestClient(app).get("/ready").status_code == 503
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/v1/models").json() == {"data": []}
    assert not _state.ready


def test_configuration_reloaded_after_restart(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    for _ in range(2):
        _state.reset()
        with TestClient(app) as client:
            assert client.get("/ready").status_code == 200
            assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"
            assert _state.gateway._default_config.budget == 3
            assert (
                _state.gateway._default_config.api_endpoint == "http://upstream:8000/v1"
            )


def test_secret_file_overrides_inline_key_without_logging(
    tmp_path, monkeypatch, caplog
):
    write_config(tmp_path, monkeypatch, api_key="inline-secret")
    key = tmp_path / "api-key"
    key.write_text("mounted-secret\n")
    monkeypatch.setenv("ITS_IAAS_API_KEY_FILE", str(key))
    with TestClient(app):
        assert _state.gateway._default_config.api_key == "mounted-secret"
    assert "mounted-secret" not in caplog.text
    assert "inline-secret" not in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "[]",
        '{"api_key":"secret-value"}',
        '{"endpoint":"http://llm/v1","model":"m","alg":"invalid","api_key":"secret-value"}',
    ],
)
def test_invalid_config_fails_closed_without_exposing_input(
    tmp_path, monkeypatch, content
):
    path = write_config(tmp_path, monkeypatch)
    path.write_text(content)
    with (
        pytest.raises(
            RuntimeError, match="Unable to load ITS startup configuration"
        ) as exc,
        TestClient(app),
    ):
        pass
    assert "secret-value" not in str(exc.value)
    assert exc.value.__suppress_context__
    assert not _state.ready


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_secret_fails_startup(tmp_path, monkeypatch, missing):
    write_config(tmp_path, monkeypatch)
    key = tmp_path / "api-key"
    if not missing:
        key.write_text("\n")
    monkeypatch.setenv("ITS_IAAS_API_KEY_FILE", str(key))
    with pytest.raises(RuntimeError), TestClient(app):
        pass
    assert not _state.ready


def test_secret_without_config_is_rejected(monkeypatch):
    monkeypatch.setenv("ITS_IAAS_API_KEY_FILE", "/secret")
    with (
        pytest.raises(RuntimeError, match="requires ITS_IAAS_CONFIG_FILE"),
        TestClient(app),
    ):
        pass
