from types import SimpleNamespace

import pytest

from surogate_eval.targets.model import APIModelTarget as ModelTarget


class FakeClient:
    """Stands in for the httpx client. Never touches the network."""

    def __init__(self, status_code=None, raises=False):
        self.status_code = status_code
        self.raises = raises
        self.calls = []

    def get(self, path, timeout=None):
        self.calls.append(path)
        if self.raises:
            raise ConnectionError("unreachable")
        return SimpleNamespace(status_code=self.status_code)

    def close(self):
        pass


def make_target(api_key, client):
    """Build a ModelTarget without running __init__ (which opens a socket)."""
    target = ModelTarget.__new__(ModelTarget)
    target.name = "t1"
    target.base_url = "https://api.openai.com/v1"
    target.api_key = api_key
    target.provider = None
    target.client = client
    return target


def test_missing_key_is_unhealthy():
    target = make_target("", FakeClient(status_code=200))
    assert target.health_check() is False


def test_rejected_credential_is_unhealthy():
    """401 is exactly the E-RUN-2 case: a key that is present but wrong."""
    target = make_target("sk-wrong", FakeClient(status_code=401))
    assert target.health_check() is False


def test_unreachable_endpoint_is_unhealthy():
    """Previously this returned True whenever a key was present."""
    target = make_target("sk-real", FakeClient(raises=True))
    assert target.health_check() is False


def test_working_endpoint_is_healthy():
    target = make_target("sk-real", FakeClient(status_code=200))
    assert target.health_check() is True


def test_placeholder_key_is_unhealthy():
    """Belt and braces: the literal ${VAR} form must never pass."""
    target = make_target("${OPENAI_API_KEY}", FakeClient(status_code=401))
    assert target.health_check() is False


def test_versioned_base_url_probes_models_first():
    """base_url already ends in /v1, so /v1/models would ask for /v1/v1/models."""
    client = FakeClient(status_code=200)
    target = make_target("sk-real", client)
    target.base_url = "https://api.openai.com/v1"
    assert target.health_check() is True
    assert client.calls == ["/models"]


def test_unversioned_base_url_probes_v1_models_first():
    client = FakeClient(status_code=200)
    target = make_target("sk-real", client)
    target.base_url = "https://api.example.com"
    assert target.health_check() is True
    assert client.calls == ["/v1/models"]


def test_local_target_probes_in_the_same_order():
    client = FakeClient(status_code=404)
    target = make_target("", client)
    target.base_url = "http://localhost:8000/v1"
    assert target.health_check() is False
    assert client.calls == ["/models", "/v1/models", "/health"]


def test_local_target_does_not_stop_at_a_401():
    """No credential is in play locally, so a 401 is just the wrong path and
    the probe carries on to /health."""
    client = FakeClient(status_code=401)
    target = make_target("", client)
    target.base_url = "http://localhost:8000/v1"
    assert target.health_check() is False
    assert client.calls == ["/models", "/v1/models", "/health"]


def test_remote_target_stops_at_a_401():
    """A key the server refuses on one path it will refuse on the next."""
    client = FakeClient(status_code=401)
    target = make_target("sk-wrong", client)
    target.base_url = "https://api.example.com"
    assert target.health_check() is False
    assert client.calls == ["/v1/models"]
