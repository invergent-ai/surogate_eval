import time
from types import SimpleNamespace

import pytest

import httpx

from surogate_eval.targets import model
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



# --- cold serverless targets -------------------------------------------


class TimeoutRecordingClient(FakeClient):
    """FakeClient that also records the timeout each probe was given."""

    def __init__(self, status_code=None, raises=False):
        super().__init__(status_code=status_code, raises=raises)
        self.timeouts = []

    def get(self, path, timeout=None):
        self.timeouts.append(timeout)
        return super().get(path, timeout=timeout)


def test_a_remote_probe_waits_long_enough_for_a_cold_start():
    """A serverless target scaled to zero is slow to ANSWER, not slow to
    connect: the platform's front door completes the handshake at once and
    holds the request while the container boots. Measured on Modal at 65s
    cold, so a 10s read budget could never catch one and a run whose model
    was merely idle was skipped entirely."""
    client = TimeoutRecordingClient(status_code=200)
    make_target("sk-real", client).health_check()

    assert client.timeouts[0].read == model.COLD_START_READ_TIMEOUT
    assert model.COLD_START_READ_TIMEOUT >= 60


def test_a_remote_probe_gives_up_on_connecting_much_sooner():
    """The other half of the same asymmetry, and what keeps the long read
    budget affordable: an unroutable or refusing host never gets past
    connect, so it costs the connect budget rather than the whole wait."""
    client = TimeoutRecordingClient(status_code=200)
    make_target("sk-real", client).health_check()

    timeout = client.timeouts[0]
    assert timeout.connect == model.CONNECT_TIMEOUT
    assert timeout.connect < timeout.read


def test_a_listening_but_silent_server_is_bounded_by_the_read_budget(monkeypatch):
    """Behavioural, against a real socket rather than a fake.

    A server that accepts the connection and never replies is exactly the
    shape of a cold start, so it must be bounded by the read budget and not
    hang the run. Constants are shrunk so the suite does not wait 90s."""
    import socket
    import threading

    # Deliberately unequal. Shrinking both to the same number made the
    # assertion below undecidable: `>= COLD_START_READ_TIMEOUT` and
    # `>= CONNECT_TIMEOUT` were the same claim, so a probe that spent the
    # wrong budget still passed.
    monkeypatch.setattr(model, "COLD_START_READ_TIMEOUT", 2)
    monkeypatch.setattr(model, "CONNECT_TIMEOUT", 1)

    # 127.0.0.2, not 127.0.0.1: health_check dispatches to its LOCAL branch
    # (a different timeout entirely) for any base_url containing "localhost",
    # "127.0.0.1" or "0.0.0.0". Addressing loopback by another octet keeps
    # this on the remote path, which is the one under test. Caught by timing
    # the test -- it passed against the local branch and proved nothing.
    listener = socket.socket()
    listener.bind(("127.0.0.2", 0))
    listener.listen(2)
    port = listener.getsockname()[1]

    # Accept every connection and say nothing, HOLDING each one open. The
    # holding is the whole point: an earlier version let the accepted socket
    # fall out of scope, which closed it, and the first probe then died of
    # "connection reset by peer" in 0.00s instead of timing out. Only the
    # second path exercised the read budget, and the test still passed.
    accepted = []

    def accept_and_hold():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:  # listener closed in the cleanup below
                return
            accepted.append(conn)

    threading.Thread(target=accept_and_hold, daemon=True).start()

    target = ModelTarget.__new__(ModelTarget)
    target.name = "silent"
    target.base_url = f"http://127.0.0.2:{port}/v1"
    target.api_key = "sk-real"
    target.provider = None
    target.client = httpx.Client(base_url=target.base_url)

    started = time.monotonic()
    try:
        assert target.health_check() is False
    finally:
        target.client.close()
        listener.close()
        for conn in accepted:
            conn.close()
    elapsed = time.monotonic() - started

    # Expected shape is COLD_START_READ_TIMEOUT on the first path and
    # CONNECT_TIMEOUT on the second, so ~3s here.
    #
    # Both bounds earn their place. The lower one says the READ budget is
    # what ended the probe -- without it the test passes even when the
    # connection dies instantly, which is how the reset-by-peer version
    # above went unnoticed. The upper one says the budgets are the shrunk
    # ones and not production values: 90 + 10 would be 100s.
    assert elapsed >= model.COLD_START_READ_TIMEOUT, f"ended early at {elapsed:.2f}s"
    assert elapsed < 8, f"probe took {elapsed:.1f}s"


def test_only_the_first_path_pays_for_a_cold_start():
    """The cold-start wait belongs once per TARGET, not once per path.

    By the time the first path has spent the full read budget, the container
    has either booted or is wedged, so giving the second path a fresh 90s
    buys nothing and doubles what a hung endpoint costs. `eval.py` probes
    targets in a plain serial loop with no deadline anywhere, so that cost
    lands N times over before evaluation starts."""
    client = TimeoutRecordingClient(status_code=404)  # nothing answers 200
    make_target("sk-real", client).health_check()

    assert len(client.timeouts) == 2, client.calls
    assert client.timeouts[0].read == model.COLD_START_READ_TIMEOUT
    assert client.timeouts[1].read == model.CONNECT_TIMEOUT
    assert client.timeouts[1].connect == model.CONNECT_TIMEOUT


def test_a_second_path_can_still_answer():
    """The allow direction: shortening the later budget must not stop a
    target whose first path simply 404s from being found on the second."""
    class SecondPathAnswers(TimeoutRecordingClient):
        def get(self, path, timeout=None):
            self.timeouts.append(timeout)
            self.calls.append(path)
            return SimpleNamespace(status_code=200 if len(self.calls) > 1 else 404)

    target = make_target("sk-real", SecondPathAnswers())
    assert target.health_check() is True
    assert len(target.client.calls) == 2


# --- The budget itself is tunable ------------------------------------------
#
# E-RUN-18. The split above is the right shape and the number in it was one
# measurement wide (Modal at 65s cold), so a slower cold start was reported
# as a target that does not exist. A budget that can only be corrected by
# rebuilding and republishing a 5 GB image is a budget nobody corrects.


def test_the_budget_can_be_raised_from_the_environment(monkeypatch):
    monkeypatch.setenv("EVAL_COLD_START_READ_TIMEOUT", "600")
    assert model._seconds_from_env("EVAL_COLD_START_READ_TIMEOUT", 300) == 600


@pytest.mark.parametrize("raw", ["", "   ", "soon", "90s", "0", "-5"])
def test_an_unusable_budget_falls_back_to_the_default(monkeypatch, raw):
    """A typo in a pod env var must not decide how long a probe waits, and
    must not take the run down at import time either. `_seconds_from_env`
    says why zero and negative are in this list."""
    monkeypatch.setenv("EVAL_COLD_START_READ_TIMEOUT", raw)
    assert model._seconds_from_env("EVAL_COLD_START_READ_TIMEOUT", 300) == 300


def test_the_default_budget_outlasts_the_one_that_was_too_short():
    """90s was hit live by a model that was merely cold. The default has to
    leave room for a container that is slower than the one it was measured
    on, and it costs nothing to do so: only a host that completes a handshake
    and then says nothing ever spends this budget."""
    assert model.COLD_START_READ_TIMEOUT > 90
