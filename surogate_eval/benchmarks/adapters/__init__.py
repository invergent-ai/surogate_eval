# Custom benchmark adapters for datasets not natively supported by evalscope.
# These register into evalscope's benchmark registry on import.

from surogate_eval.benchmarks.adapters.deep_planning import DeepPlanningAdapter
from surogate_eval.benchmarks.adapters.wide_search import WideSearchAdapter
from surogate_eval.benchmarks.adapters.mcp_atlas import MCPAtlasAdapter
from surogate_eval.benchmarks.adapters.vita_bench import VitaBenchAdapter
from surogate_eval.benchmarks.adapters.swe_bench_pro import SWEBenchProAdapter
from surogate_eval.benchmarks.adapters.swe_bench_multilingual import SWEBenchMultilingualAdapter
