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


class SlowStartClient:
    """Times out on the first *n* probes, then answers.

    A serverless target that has scaled to zero behaves exactly like this:
    the probe times out while the container boots, and a later one succeeds.
    Measured against a Modal endpoint: 65s cold, 1.07s warm.
    """

    def __init__(self, timeouts):
        self.timeouts = timeouts
        self.calls = []

    def get(self, path, timeout=None):
        self.calls.append((path, timeout))
        if len(self.calls) <= self.timeouts:
            raise ConnectionError("timed out")
        return SimpleNamespace(status_code=200)

    def close(self):
        pass


def test_a_target_that_wakes_up_late_is_healthy():
    """The first pass cannot outlast a cold start, so failing there must not
    be the final word. Both first-pass paths time out here; the target
    answers after that."""
    target = make_target("sk-real", SlowStartClient(timeouts=2))
    assert target.health_check() is True


def test_the_retry_waits_longer_than_the_first_pass():
    """A retry on the same budget would time out identically and only cost
    another 10s. It has to allow enough time for the target to boot."""
    client = SlowStartClient(timeouts=2)
    target = make_target("sk-real", client)
    target.health_check()

    first_pass = [t for _, t in client.calls[:2]]
    retry = [t for _, t in client.calls[2:]]
    assert retry, "no retry was attempted"
    assert min(retry) > max(first_pass)


def test_an_endpoint_that_never_answers_is_still_unhealthy():
    """The allow direction: patience must not make a dead endpoint healthy."""
    target = make_target("sk-real", SlowStartClient(timeouts=99))
    assert target.health_check() is False
