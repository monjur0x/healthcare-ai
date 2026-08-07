"""
CSV validation for structured healthcare data.

Validates file type, structure, and required columns before any
downstream preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import settings
from ..exceptions import (
    EmptyDatasetError,
    InvalidCSVError,
    MissingColumnError,
    UnsupportedFileTypeError,
)
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable result of a CSV validation run.
    """

    is_valid: bool
    dataframe: pd.DataFrame
    total_rows: int
    total_columns: int
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


class CSVValidator:
    """
    Validates CSV files and returns a validated dataframe.

    The validator is the first stage of the CSV pipeline. It enforces
    supported file types, non-empty datasets, and presence of required
    columns.

    Parameters
    ----------
    required_columns : tuple[str, ...]
        Columns that must exist in the CSV file. Defaults to empty.
    """

    def __init__(self, required_columns: tuple[str, ...] = ()) -> None:
        self._required_columns = tuple(required_columns)

    @staticmethod
    def is_supported(path: str | Path) -> bool:
        """
        Check whether a file path has a supported CSV extension.

        Parameters
        ----------
        path : str | Path
            Path to inspect.

        Returns
        -------
        bool
            True if the file extension is in SUPPORTED_CSV_TYPES.
        """

        return Path(path).suffix.lower() in settings.SUPPORTED_CSV_TYPES

    def validate_file(self, path: str | Path) -> ValidationResult:
        """
        Validate a CSV file on disk and read it into a dataframe.

        Parameters
        ----------
        path : str | Path
            Path to the CSV file.

        Returns
        -------
        ValidationResult
            Validated dataframe plus validation metadata.

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not a supported CSV type.
        InvalidCSVError
            If the file cannot be parsed as CSV.
        EmptyDatasetError
            If the file contains no rows.
        """

        path = Path(path)

        if not self.is_supported(path):
            logger.error("Unsupported file type: %s", path)
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported types: {settings.SUPPORTED_CSV_TYPES}."
            )

        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:
            logger.error("Failed to parse CSV file %s: %s", path, exc)
            raise InvalidCSVError(f"Could not parse CSV file '{path}': {exc}") from exc

        return self.validate_dataframe(dataframe)

    def validate_dataframe(self, dataframe: pd.DataFrame) -> ValidationResult:
        """
        Validate an in-memory dataframe.

        This method allows validation of dataframes built from uploaded
        bytes without a file on disk.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe to validate.

        Returns
        -------
        ValidationResult
            Validated dataframe plus validation metadata.

        Raises
        ------
        EmptyDatasetError
            If the dataframe contains no rows.
        MissingColumnError
            If required columns are absent.
        """

        if dataframe.empty:
            logger.error("Dataset is empty")
            raise EmptyDatasetError("Dataset contains no rows.")

        present = set(dataframe.columns)
        missing = tuple(col for col in self._required_columns if col not in present)

        warnings: list[str] = []
        if missing:
            warnings.append(f"Missing required columns: {', '.join(missing)}")
            logger.warning("Missing required columns: %s", missing)
            raise MissingColumnError(f"Missing required columns: {', '.join(missing)}")

        result = ValidationResult(
            is_valid=True,
            dataframe=dataframe.copy(),
            total_rows=len(dataframe),
            total_columns=dataframe.shape[1],
            required_columns=self._required_columns,
            missing_columns=missing,
            warnings=tuple(warnings),
        )
        logger.info(
            "Validated CSV: %d rows, %d columns",
            result.total_rows,
            result.total_columns,
        )
        return result
