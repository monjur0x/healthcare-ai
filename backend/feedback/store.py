"""
Persistent store for clinician feedback samples.

The store records confirmed outcome labels for past analyses so a model
can be retrained on real-world corrections. Rows carry a ``consumed``
flag: samples folded into a retrain are marked consumed so they are not
re-used by a later retrain.
"""

from __future__ import annotations

import json
import sqlite3
import threading

from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .schemas import FeedbackRecord


class FeedbackStoreError(Exception):
    """Raised when a feedback store operation fails."""


class FeedbackStore:
    """
    SQLite-backed feedback store.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file. Defaults to
        ``settings.DB_PATH``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # RLock: FastAPI serves concurrent threads off one instance.
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Open (and create) the SQLite database with the feedback table."""
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path, check_same_thread=False, timeout=10.0
                )
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS feedback_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preset TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    features TEXT NOT NULL,
                    confirmed_label INTEGER NOT NULL,
                    predicted_label INTEGER,
                    confidence REAL,
                    consumed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
                )
                self._connection.commit()
            return self._connection

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def add(
        self,
        preset: str,
        patient_id: str,
        features: dict[str, float],
        confirmed_label: int,
        predicted_label: int | None = None,
        confidence: float | None = None,
    ) -> FeedbackRecord:
        """
        Persist one feedback sample and return the stored record.

        Parameters
        ----------
        preset : str
            Dataset preset the feedback refers to.
        patient_id : str
            Patient study id.
        features : dict[str, float]
            Feature row that was analyzed.
        confirmed_label : int
            Clinician-confirmed outcome label (0/1).
        predicted_label : int | None
            Model prediction at analysis time (when recorded).
        confidence : float | None
            Model confidence at analysis time (when recorded).

        Returns
        -------
        FeedbackRecord
            The persisted record (including its id and timestamp).

        Raises
        ------
        FeedbackStoreError
            If the row cannot be persisted.
        """

        try:
            with self._lock:
                cursor = self.connect().execute(
                    """
                INSERT INTO feedback_samples (
                    preset, patient_id, features, confirmed_label,
                    predicted_label, confidence, consumed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                """,
                    (
                        preset,
                        patient_id,
                        json.dumps(features),
                        int(confirmed_label),
                        predicted_label,
                        confidence,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                self._connection.commit()
                return self.get(int(cursor.lastrowid))
        except sqlite3.Error as error:
            raise FeedbackStoreError(f"Could not persist feedback: {error}") from error

    def get(self, sample_id: int) -> FeedbackRecord:
        """
        Return a single feedback record by id.

        Parameters
        ----------
        sample_id : int
            Row id.

        Returns
        -------
        FeedbackRecord
            The stored record.

        Raises
        ------
        FeedbackStoreError
            If the row does not exist.
        """

        with self._lock:
            row = (
                self.connect()
                .execute(
                    """
            SELECT id, preset, patient_id, features, confirmed_label,
                   predicted_label, confidence, created_at, consumed
            FROM feedback_samples
            WHERE id = ?
            """,
                    (sample_id,),
                )
                .fetchone()
            )
        if row is None:
            raise FeedbackStoreError(f"No feedback row with id {sample_id}.")
        return self._row_to_record(row)

    def list_pending(self, preset: str) -> list[FeedbackRecord]:
        """
        Return unconsumed feedback samples for a preset (oldest first).

        Parameters
        ----------
        preset : str
            Dataset preset.

        Returns
        -------
        list[FeedbackRecord]
            Pending samples ordered by creation time.
        """

        with self._lock:
            rows = (
                self.connect()
                .execute(
                    """
            SELECT id, preset, patient_id, features, confirmed_label,
                   predicted_label, confidence, created_at, consumed
            FROM feedback_samples
            WHERE preset = ? AND consumed = 0
            ORDER BY created_at ASC
            """,
                    (preset,),
                )
                .fetchall()
            )
        return [self._row_to_record(row) for row in rows]

    def count_pending(self, preset: str) -> int:
        """Return the number of unconsumed samples for a preset."""
        with self._lock:
            row = (
                self.connect()
                .execute(
                    "SELECT COUNT(*) FROM feedback_samples "
                    "WHERE preset = ? AND consumed = 0",
                    (preset,),
                )
                .fetchone()
            )
        return int(row[0])

    def recent(self, preset: str, limit: int = 5) -> list[FeedbackRecord]:
        """
        Return the most recent samples for a preset (newest first).

        Parameters
        ----------
        preset : str
            Dataset preset.
        limit : int
            Maximum number of rows.

        Returns
        -------
        list[FeedbackRecord]
            Recent samples ordered by creation time (descending).
        """

        with self._lock:
            rows = (
                self.connect()
                .execute(
                    """
            SELECT id, preset, patient_id, features, confirmed_label,
                   predicted_label, confidence, created_at, consumed
            FROM feedback_samples
            WHERE preset = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
                    (preset, limit),
                )
                .fetchall()
            )
        return [self._row_to_record(row) for row in rows]

    def mark_consumed(self, sample_ids: list[int]) -> int:
        """
        Mark feedback samples as consumed by a retrain.

        Parameters
        ----------
        sample_ids : list[int]
            Row ids to mark consumed.

        Returns
        -------
        int
            Number of rows updated.
        """

        if not sample_ids:
            return 0
        placeholders = ",".join("?" for _ in sample_ids)
        query = (
            "UPDATE feedback_samples SET consumed = 1 "
            f"WHERE id IN ({placeholders}) AND consumed = 0"
        )
        with self._lock:
            cursor = self.connect().execute(query, sample_ids)
            self._connection.commit()
        return int(cursor.rowcount)

    @staticmethod
    def _row_to_record(row: sqlite3.Row | tuple) -> FeedbackRecord:
        """Convert a SQLite row to a ``FeedbackRecord``."""
        return FeedbackRecord(
            id=int(row[0]),
            preset=row[1],
            patient_id=row[2],
            features=json.loads(row[3]) if row[3] else {},
            confirmed_label=int(row[4]),
            predicted_label=int(row[5]) if row[5] is not None else None,
            confidence=float(row[6]) if row[6] is not None else None,
            created_at=row[7],
            consumed=bool(row[8]),
        )


__all__ = ["FeedbackStore", "FeedbackStoreError"]
