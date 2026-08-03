"""Configuring a judge must not discard the benchmark's other settings.

``GenericBenchmark.evaluate`` builds the dict it hands to the backend, and
it already puts the whole of ``backend_params`` in. A later block replaced
that entry with ``{'judge_target': ...}`` whenever a judge was configured,
so every other key the backend reads - ``use_sandbox``, ``max_turns``,
``timeout_multiplier``, ``batch_size``, ``max_workers``,
``generation_config`` - vanished the moment a benchmark was judge-scored.
There was no warning: the benchmark ran on defaults and reported success.

The replacement was never needed. ``runners.py`` puts ``judge_target``
*into* ``backend_params`` before the benchmark runs, so the dict passed
earlier in the same function already carried it.

``test_judge_scored_benchmark_reaches_evalscope_with_its_turn_limit``
closes the loop, because asserting the dict survives is not enough on its
own. The evalscope backend reads ``max_turns`` inside its ``if
judge_target:`` branch, and the wipe put that value out of reach by
construction: the presence of the judge was exactly what emptied the dict
the lookup reads. So the config ``GenericBenchmark`` produces is fed to the
real ``_prepare_task_config`` and checked for the value that could never
arrive.

No network: the backend is a stub for the passthrough tests, and
``_prepare_task_config`` is pure config assembly.
"""

from surogate_eval.benchmarks.base import BenchmarkConfig
from surogate_eval.benchmarks.generic import GenericBenchmark

BACKEND_PARAMS = {
    "use_sandbox": False,
    "max_turns": 200,
    "timeout_multiplier": 3.0,
    "batch_size": 8,
    "max_workers": 4,
}


class FakeJudge:
    config = {"base_url": "https://judge.example/v1", "model": "jm", "api_key": "jk"}


class FakeTarget:
    name = "t1"
    config = {"model": "m", "base_url": "https://target.example", "api_key": "k"}


class CapturingBackend:
    """Records the config it was handed instead of running an evaluation."""

    def __init__(self):
        self.config = None

    def evaluate(self, target, benchmark_name, config):
        self.config = config
        return {"overall_score": 0.0, "task_results": {}, "detailed_results": []}


def _run() -> tuple[dict, object]:
    """Drive the real ``evaluate``.

    Returns the config the backend was handed, plus the real evalscope
    backend the benchmark built for itself, so a test can drive that rather
    than hand-rolling a second one.
    """
    # runners.py puts the resolved judge into backend_params before the
    # benchmark runs, so this is the state evaluate() actually sees.
    params = dict(BACKEND_PARAMS, judge_target=FakeJudge())

    benchmark = GenericBenchmark(
        BenchmarkConfig(name="tau_bench", backend="evalscope", backend_params=params)
    )
    real_backend = benchmark.backend

    capturing = CapturingBackend()
    benchmark.backend = capturing
    benchmark.evaluate(FakeTarget())

    return capturing.config, real_backend


def test_backend_params_survive_a_configured_judge():
    passed, _ = _run()

    for key, value in BACKEND_PARAMS.items():
        assert passed["backend_params"][key] == value, f"{key} was dropped"


def test_judge_target_still_reaches_the_backend():
    """The allow direction: not dropping the others must not drop the judge."""
    passed, _ = _run()

    assert isinstance(passed["backend_params"]["judge_target"], FakeJudge)


def test_judge_scored_benchmark_reaches_evalscope_with_its_turn_limit():
    """The settings must survive as far as evalscope's ``dataset_args``.

    ``max_num_steps`` is the assertion that proves the fix: evalscope writes
    it only inside its own ``if judge_target:`` branch, the one the wipe put
    out of reach. ``timeout_multiplier`` covers the unconditional
    ``extra_params`` path beside it.
    """
    config, backend = _run()

    task_config = backend._prepare_task_config(FakeTarget(), "tau_bench", config)

    extra = task_config.dataset_args["tau_bench"]["extra_params"]
    assert extra["max_num_steps"] == BACKEND_PARAMS["max_turns"]
    assert extra["timeout_multiplier"] == BACKEND_PARAMS["timeout_multiplier"]
