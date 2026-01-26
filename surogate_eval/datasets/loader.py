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

    def download_tokenizer_dir(self, lakefs_uri: str) -> str:
        """
        Download all tokenizer files from LakeFS to a temp directory.

        Args:
            lakefs_uri: LakeFS path (can point to a file or directory)
                        e.g., lakefs://repo/branch/tokenizer.json or lakefs://repo/branch

        Returns:
            Path to temp directory containing tokenizer files
        """
        import shutil

        # Get base path (remove filename if present)
        if lakefs_uri.endswith('.json') or lakefs_uri.endswith('.txt'):
            base_path = lakefs_uri.rsplit('/', 1)[0]
        else:
            base_path = lakefs_uri

        # Create temp directory for tokenizer files
        temp_dir = tempfile.mkdtemp(prefix="tokenizer_")

        # Files needed for tokenizer
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt',
            'special_tokens_map.json',
            'added_tokens.json',
        ]

        downloaded = 0
        for filename in tokenizer_files:
            try:
                lakefs_path = f"{base_path}/{filename}"
                local_path = self._download_from_lakefs(lakefs_path)
                shutil.copy(local_path, os.path.join(temp_dir, filename))
                downloaded += 1
                logger.debug(f"Downloaded tokenizer file: {filename}")
            except Exception as e:
                logger.debug(f"Optional tokenizer file {filename} not found: {e}")

        if downloaded == 0:
            shutil.rmtree(temp_dir)
            raise FileNotFoundError(f"No tokenizer files found at {base_path}")

        logger.info(f"Downloaded {downloaded} tokenizer files to: {temp_dir}")
        return temp_dir

    def _download_from_lakefs(self, lakefs_uri: str) -> str:
        """
        Download file from LakeFS to local temp path.
        If no file path specified, auto-discovers single dataset file.

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

        # Parse lakefs://repo/branch or lakefs://repo/branch/path
        uri_parts = lakefs_uri.replace('lakefs://', '').split('/', 2)
        if len(uri_parts) < 2:
            raise ValueError(f"Invalid LakeFS URI: {lakefs_uri}. Expected lakefs://repo/branch[/path]")

        repo = uri_parts[0]
        branch = uri_parts[1]
        path = uri_parts[2] if len(uri_parts) > 2 else None

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

        repo_obj = lakefs.Repository(repo, client=client)
        branch_obj = repo_obj.branch(branch)

        # Auto-discover if no path given
        if not path:
            path = self._discover_dataset_file(branch_obj, lakefs_uri)

        logger.info(f"Downloading from LakeFS: {lakefs_uri} -> {path}")

        # Download to temp file
        suffix = Path(path).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()

        try:
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

    def _discover_dataset_file(self, branch_obj, lakefs_uri: str) -> str:
        """
        Find the single dataset file in the branch.

        Raises error if zero or multiple dataset files found.
        """
        dataset_files = []

        for obj in branch_obj.objects():
            path = obj.path
            suffix = Path(path).suffix.lower()
            if suffix in self.SUPPORTED_FORMATS:
                dataset_files.append(path)

        if len(dataset_files) == 0:
            raise FileNotFoundError(
                f"No dataset files found in {lakefs_uri}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        if len(dataset_files) > 1:
            raise ValueError(
                f"Multiple dataset files found in {lakefs_uri}: {dataset_files}. "
                f"Please specify the file path explicitly."
            )

        logger.info(f"Auto-discovered dataset file: {dataset_files[0]}")
        return dataset_files[0]


    def load(self, filepath: str, format: Optional[str] = None, limit: Optional[int] = None) -> pl.DataFrame:
        """
        Load dataset from file.

        Args:
            filepath: Path to dataset file (local or lakefs://)
            format: File format (auto-detected if None)
            limit: Maximum number of rows to load

        Returns:
            Polars DataFrame
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

        logger.info(f"Loading dataset from {filepath} (format: {format}, limit: {limit})")

        if format == '.jsonl':
            df = self._load_jsonl(path)
        elif format == '.csv':
            df = self._load_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Apply limit
        if limit is not None and limit > 0:
            df = df.head(limit)

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def detect_dataset_type(self, path: str) -> str:
        """
        Detect if dataset is single-turn or multi-turn.
        Only loads first row to check columns.
        """
        df = self.load(path, limit=1)  # Just need 1 row to check columns

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

    def load_test_cases(self, filepath: str, limit: Optional[int] = None) -> List[Union[TestCase, MultiTurnTestCase]]:
        """
        Load test cases from file.

        Args:
            filepath: Path to dataset file (local or lakefs://)
            limit: Maximum number of test cases to load

        Returns:
            List of TestCase or MultiTurnTestCase objects
        """
        df = self.load(filepath, limit=limit)

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