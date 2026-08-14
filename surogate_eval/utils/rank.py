"""Which process may write the run's artifacts.

Stdlib only, deliberately. ``utils/dist.py`` answers the same question, but
it imports torch, transformers and datasets at module level, and the write
paths here run in installs where torch is optional (``cli/main.py`` guards
its own torch import for exactly that reason). A results file must not
depend on a deep-learning stack being importable.
"""

import os


def is_rank_zero() -> bool:
    """True in a single-process run, and on rank 0 of a distributed one.

    ``RANK`` unset reads as -1, which is the single-process case and the only
    one that runs today: eval pods request no GPUs, so ``cli/main.py`` never
    takes its ``torch.distributed.run`` branch. That changes the moment an
    eval pod gets GPUs (the colocated judge), which is why the guard is here
    before the condition exists rather than after.
    """
    return int(os.environ.get("RANK", -1)) in {-1, 0}
