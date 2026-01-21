# surogate/eval/benchmarks/registry.py
"""Registry for all available benchmarks."""

from typing import Dict, Type, List, Optional
from .base import BaseBenchmark, BenchmarkConfig
from ..utils.logger import get_logger

logger = get_logger()


class BenchmarkRegistry:
    """Registry for benchmark implementations."""

    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a benchmark.

        Args:
            name: Benchmark name
        """

        def decorator(benchmark_class: Type[BaseBenchmark]):
            cls._benchmarks[name] = benchmark_class
            logger.debug(f"Registered benchmark: {name}")
            return benchmark_class

        return decorator

    @classmethod
    def create_benchmark(cls, config: BenchmarkConfig) -> BaseBenchmark:
        benchmark_name = config.name

        # If benchmark is registered, use it
        if benchmark_name in cls._benchmarks:
            benchmark_class = cls._benchmarks[benchmark_name]
            return benchmark_class(config)

        # Otherwise, use GenericBenchmark (lm-eval/evalscope handles validation)
        from .generic import GenericBenchmark
        logger.debug(f"Using GenericBenchmark for '{benchmark_name}'")
        return GenericBenchmark(config)

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """Get list of registered benchmarks."""
        return list(cls._benchmarks.keys())

    @classmethod
    def get_benchmark_info(cls, name: str) -> Optional[Dict]:
        """
        Get information about a benchmark.

        Args:
            name: Benchmark name

        Returns:
            Benchmark information or None
        """
        if name not in cls._benchmarks:
            return None

        benchmark_class = cls._benchmarks[name]
        return {
            'name': name,
            'class': benchmark_class.__name__,
            'doc': benchmark_class.__doc__,
        }