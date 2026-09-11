"""
Persistent store for assembled clinical reports.

The n8n "Store Results" step (proposal §10, step 8) persists the
assembled pipeline report here so results survive beyond the webhook
response. Mirrors the locking and journal configuration of the risk
and feedback stores: one ``RLock``-guarded SQLite connection with
``check_same_thread=False``, a 10s busy timeout, and WAL journal mode.
"""

from __future__ import annotations

import json
import sqlite3
import threading

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class ReportStoreError(Exception):
    """Raised when a report cannot be persisted or retrieved."""


@dataclass
class ReportRecord:
    """One persisted clinical report."""

    report_id: int
    patient_id: str
    preset: str
    report: dict
    stored_at: str


class ReportStore:
    """
    SQLite-backed clinical report store.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file. Parent directories are
        created on demand.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # RLock: FastAPI serves concurrent threads off one instance.
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Open (and create) the SQLite database with the reports table."""
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path, check_same_thread=False, timeout=10.0
                )
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    report TEXT NOT NULL,
                    stored_at TEXT NOT NULL
                )
                """
                )
                self._connection.execute(
                    """
                CREATE INDEX IF NOT EXISTS idx_reports_patient
                ON reports (patient_id, stored_at)
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
        patient_id: str,
        preset: str,
        report: dict,
    ) -> ReportRecord:
        """
        Persist one assembled report.

        Parameters
        ----------
        patient_id : str
            Patient study id (never PHI).
        preset : str
            Dataset preset the report was produced under.
        report : dict
            The assembled report payload (must be JSON-serializable).

        Returns
        -------
        ReportRecord
            The persisted record, including its id and timestamp.

        Raises
        ------
        ReportStoreError
            If the payload is not JSON-serializable or the row cannot
            be persisted.
        """
        try:
            payload = json.dumps(report)
        except (TypeError, ValueError) as error:
            raise ReportStoreError(
                f"Report payload is not JSON-serializable: {error}"
            ) from error
        stored_at = datetime.now(UTC).isoformat()
        try:
            with self._lock:
                cursor = self.connect().execute(
                    """
                INSERT INTO reports (patient_id, preset, report, stored_at)
                VALUES (?, ?, ?, ?)
                """,
                    (patient_id, preset, payload, stored_at),
                )
                self._connection.commit()
                return ReportRecord(
                    report_id=int(cursor.lastrowid),
                    patient_id=patient_id,
                    preset=preset,
                    report=report,
                    stored_at=stored_at,
                )
        except sqlite3.Error as error:
            raise ReportStoreError(f"Could not persist report: {error}") from error

    def get(self, report_id: int) -> ReportRecord | None:
        """
        Retrieve one report by id, or None when it does not exist.
        """
        with self._lock:
            row = (
                self.connect()
                .execute(
                    """
            SELECT id, patient_id, preset, report, stored_at
            FROM reports WHERE id = ?
            """,
                    (report_id,),
                )
                .fetchone()
            )
        if row is None:
            return None
        return ReportRecord(
            report_id=int(row[0]),
            patient_id=str(row[1]),
            preset=str(row[2]),
            report=json.loads(str(row[3])),
            stored_at=str(row[4]),
        )
