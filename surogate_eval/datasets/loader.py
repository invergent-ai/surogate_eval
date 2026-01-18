from pathlib import Path
from typing import Optional, List, Union
import polars as pl
import json
import os
import tempfile

from .test_case import TestCase, MultiTurnTestCase, Turn
from ..utils.logger import get_logger

logger = get_logger()


class DatasetLoader:
    """Load datasets from various formats using Polars."""

    SUPPORTED_FORMATS = ['.jsonl', '.csv']

    def __init__(self):
        """Initialize dataset loader."""
        self._lakefs_cache = {}

    def _is_lakefs_path(self, filepath: str) -> bool:
        """Check if path is a LakeFS URI."""
        return filepath.startswith('lakefs://')

    def _download_from_lakefs(self, lakefs_uri: str) -> str:
        """
        Download file from LakeFS to local temp path.

        Returns local path to downloaded file.
        """
        if lakefs_uri in self._lakefs_cache:
            cached = self._lakefs_cache[lakefs_uri]
            if Path(cached).exists():
                logger.info(f"Using cached LakeFS file: {cached}")
                return cached

        try:
            import lakefs
        except ImportError:
            raise ImportError(
                "lakefs package required for LakeFS support. "
                "Install with: pip install lakefs"
            )

        # Parse lakefs://repo/branch/path
        uri_parts = lakefs_uri.replace('lakefs://', '').split('/', 2)
        if len(uri_parts) < 3:
            raise ValueError(f"Invalid LakeFS URI: {lakefs_uri}. Expected lakefs://repo/branch/path")

        repo, branch, path = uri_parts[0], uri_parts[1], uri_parts[2]

        logger.info(f"Downloading from LakeFS: {lakefs_uri}")

        # Configure client from environment
        endpoint = os.environ.get('LAKECTL_SERVER_ENDPOINT_URL') or os.environ.get('LAKEFS_ENDPOINT')
        access_key = os.environ.get('LAKECTL_CREDENTIALS_ACCESS_KEY_ID') or os.environ.get('LAKEFS_KEY')
        secret_key = os.environ.get('LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY') or os.environ.get('LAKEFS_SECRET')

        if not all([endpoint, access_key, secret_key]):
            raise ValueError(
                "LakeFS credentials not configured. Set environment variables: "
                "LAKEFS_ENDPOINT, LAKEFS_KEY, LAKEFS_SECRET"
            )

        client = lakefs.Client(
            host=endpoint,
            username=access_key,
            password=secret_key
        )

        # Download to temp file
        suffix = Path(path).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()

        try:
            repo_obj = lakefs.Repository(repo, client=client)
            branch_obj = repo_obj.branch(branch)
            obj = branch_obj.object(path)

            with open(temp_path, 'wb') as f:
                for chunk in obj.reader():
                    f.write(chunk)

            logger.info(f"Downloaded LakeFS file to: {temp_path}")
            self._lakefs_cache[lakefs_uri] = temp_path
            return temp_path

        except Exception as e:
            if Path(temp_path).exists():
                os.unlink(temp_path)
            raise FileNotFoundError(f"Failed to download from LakeFS: {lakefs_uri}. Error: {e}")

    def load(self, filepath: str, format: Optional[str] = None) -> pl.DataFrame:
        """
        Load dataset from file.

        Args:
            filepath: Path to dataset file (local or lakefs://)
            format: File format (auto-detected if None)

        Returns:
            Polars DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        # Handle LakeFS paths
        if self._is_lakefs_path(filepath):
            local_path = self._download_from_lakefs(filepath)
            path = Path(local_path)
        else:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {filepath}")

        # Auto-detect format
        if format is None:
            format = path.suffix.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")

        logger.info(f"Loading dataset from {filepath} (format: {format})")

        if format == '.jsonl':
            df = self._load_jsonl(path)
        elif format == '.csv':
            df = self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def detect_dataset_type(self, path: str) -> str:
        """
        Detect if dataset is single-turn or multi-turn.

        Args:
            path: Path to dataset file (local or lakefs://)

        Returns:
            'single_turn' or 'multi_turn'
        """
        df = self.load(path)

        if 'turns' in df.columns:
            return 'multi_turn'

        if 'messages' in df.columns or 'conversation' in df.columns:
            return 'multi_turn'

        if 'input' in df.columns:
            return 'single_turn'

        logger.warning("Could not definitively detect dataset type, defaulting to single_turn")
        return 'single_turn'

    def _load_jsonl(self, path: Path) -> pl.DataFrame:
        """Load JSONL file."""
        try:
            df = pl.read_ndjson(path)
            return df
        except Exception as e:
            logger.error(f"Failed to load JSONL with Polars: {e}")
            return self._load_jsonl_manual(path)

    def _load_jsonl_manual(self, path: Path) -> pl.DataFrame:
        """Manually parse JSONL for complex structures."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        if not data:
            raise ValueError("No valid data found in JSONL file")

        df = pl.DataFrame(data)
        return df

    def _load_csv(self, path: Path) -> pl.DataFrame:
        """Load CSV file."""
        try:
            df = pl.read_csv(path)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

    def load_test_cases(self, filepath: str) -> List[Union[TestCase, MultiTurnTestCase]]:
        """
        Load test cases from file.

        Args:
            filepath: Path to dataset file (local or lakefs://)

        Returns:
            List of TestCase or MultiTurnTestCase objects
        """
        df = self.load(filepath)

        is_multi_turn = 'turns' in df.columns

        test_cases = []
        for row in df.iter_rows(named=True):
            try:
                if is_multi_turn:
                    test_case = self._row_to_multi_turn_test_case(row)
                else:
                    test_case = self._row_to_test_case(row)
                test_cases.append(test_case)
            except Exception as e:
                logger.warning(f"Failed to convert row to test case: {e}")
                continue

        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases

    def _row_to_test_case(self, row: dict) -> TestCase:
        """Convert DataFrame row to TestCase."""
        input_text = row.get('input')
        expected_output = row.get('expected_output')

        if 'metadata' in row and row['metadata']:
            metadata = row['metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata as JSON: {metadata}")
                    metadata = {}
        else:
            metadata = {k: v for k, v in row.items()
                        if k not in ['input', 'expected_output']}

        return TestCase(
            input=input_text,
            expected_output=expected_output,
            metadata=metadata
        )

    def _row_to_multi_turn_test_case(self, row: dict) -> MultiTurnTestCase:
        """Convert DataFrame row to MultiTurnTestCase."""
        turns_data = row.get('turns', [])
        expected_final_output = row.get('expected_final_output')

        turns = [Turn.from_dict(turn_dict) for turn_dict in turns_data]

        metadata = {k: v for k, v in row.items()
                    if k not in ['turns', 'expected_final_output']}

        return MultiTurnTestCase(
            turns=turns,
            expected_final_output=expected_final_output,
            metadata=metadata
        )

    def save(self, df: pl.DataFrame, filepath: str, format: Optional[str] = None):
        """
        Save dataset to file.

        Args:
            df: Polars DataFrame
            filepath: Output file path
            format: File format (auto-detected if None)
        """
        path = Path(filepath)

        if format is None:
            format = path.suffix.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saving dataset to {filepath} (format: {format})")

        if format == '.jsonl':
            df.write_ndjson(path)
        elif format == '.csv':
            df.write_csv(path)

        logger.info(f"Saved dataset: {len(df)} rows")