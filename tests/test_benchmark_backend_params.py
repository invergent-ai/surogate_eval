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


# --- keys the YAML never set ------------------------------------------


def _run_minimal(**config_kwargs) -> dict:
    """Drive the real ``evaluate`` for a benchmark that sets almost nothing."""
    benchmark = GenericBenchmark(
        BenchmarkConfig(
            name="custom-support-qa", backend="custom_eval", **config_kwargs
        )
    )
    capturing = CapturingBackend()
    benchmark.backend = capturing
    benchmark.evaluate(FakeTarget())
    return capturing.config


def test_an_unset_key_is_absent_rather_than_present_as_none():
    """A backend reads its own defaults with ``config.get(key, default)``,
    and that returns the default only when the key is ABSENT. Handing over
    every key with `None` for the ones the YAML omitted made every such
    default unreachable code.

    That is what made a judge benchmark with no criteria error all its rows:
    `config.get('judge_criteria', <text>)` returned `None`, and GEval
    refuses to build without criteria. `max_tokens` is the same shape.

    (`eval_type` is not: `BenchmarkConfig.eval_type` is a plain `str` with a
    default, so it is never None and never dropped. Its backend-side
    `.get('eval_type', 'exact_match')` still cannot fire -- harmless, since
    the two defaults agree.)"""
    passed = _run_minimal()

    nones = sorted(k for k, v in passed.items() if v is None)
    assert nones == [], f"handed to the backend as None: {nones}"


def test_a_backend_default_can_actually_fire():
    """States the consequence directly, in the form a backend uses."""
    passed = _run_minimal()

    assert passed.get("eval_type", "exact_match") == "exact_match"
    assert passed.get("max_tokens", 256) == 256
    assert passed.get("judge_criteria", "fallback") == "fallback"


def test_a_key_the_yaml_did_set_still_arrives():
    """The allow direction: dropping unset keys must not drop set ones,
    including falsy values a benchmark deliberately chose."""
    passed = _run_minimal(eval_type="judge", max_tokens=0, num_fewshot=0)

    assert passed["eval_type"] == "judge"
    assert passed["max_tokens"] == 0
    assert passed["num_fewshot"] == 0


def test_a_custom_dataset_path_turns_the_progress_tracker_off():
    """evalscope's tracker total describes the wrong dataset here.

    ``compute_eval_total_count`` (evalscope 1.7.0,
    ``utils/resource_utils.py``) sums per-subset ``sample_count`` out of the
    *bundled* ``_meta`` files and reads ``dataset_args['subset_list']``, but
    never ``dataset_args['dataset_id']`` -- which is exactly the key a
    ``path:``/``dataset_path:`` override sets. So the tracker would publish
    the size of the upstream dataset while the run evaluates a different
    one: a smaller custom dataset stalls the bar partway forever, a larger
    one has rows_done sail past rows_total.

    With the tracker off, no ``<work_dir>/progress.json`` is written,
    ``read_evalscope_tracker`` finds nothing, and the watcher publishes
    ``rows_total=0`` -- the "unknown" sentinel both repos already fall back
    on. ``rows_done`` still comes from the reviews line count, so live
    progress survives; only the unknowable total is withheld.
    """
    config, backend = _run()

    without_path = backend._prepare_task_config(FakeTarget(), "tau_bench", config)
    assert without_path.enable_progress_tracker is True, (
        "sanity: a bundled dataset must keep the tracker, or this proves nothing"
    )

    with_path = backend._prepare_task_config(
        FakeTarget(), "tau_bench", dict(config, dataset_path="/data/mine.jsonl"),
    )
    assert with_path.enable_progress_tracker is False
