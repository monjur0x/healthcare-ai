"""
SQLite model registry for federated learning.

Persists every distributed federation run: the run metadata (preset,
number of hospitals, rounds), the per-round global metrics, and the
resulting global model artifact with a monotonic version number. The
registry is the source of truth the API and dashboard use to discover
trained global models.
"""

from __future__ import annotations

import sqlite3
import uuid

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(UTC).isoformat()


class ModelRegistry:
    """
    SQLite-backed registry of federated model runs.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file (created if missing).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the registry schema if it does not exist yet."""
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                preset TEXT NOT NULL,
                n_hospitals INTEGER NOT NULL,
                n_rounds INTEGER NOT NULL,
                secure_aggregation INTEGER NOT NULL,
                differential_privacy INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS rounds (
                run_id TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                accuracy REAL,
                log_loss REAL,
                n_clients INTEGER,
                bytes_exchanged INTEGER,
                duration_s REAL,
                PRIMARY KEY (run_id, round_index),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                preset TEXT NOT NULL,
                version INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                accuracy REAL,
                roc_auc REAL,
                epsilon REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            """
        )
        self._connection.commit()

    def start_run(
        self,
        preset: str,
        n_hospitals: int,
        n_rounds: int,
        secure_aggregation: bool,
        differential_privacy: bool,
    ) -> str:
        """
        Register a new federation run and return its id.

        Parameters
        ----------
        preset : str
            Dataset preset being federated.
        n_hospitals : int
            Number of participating hospital sites.
        n_rounds : int
            Number of federated rounds.
        secure_aggregation : bool
            Whether secure aggregation was enabled.
        differential_privacy : bool
            Whether DP-SGD was enabled on the clients.

        Returns
        -------
        str
            The run's unique id.
        """

        run_id = uuid.uuid4().hex[:12]
        self._connection.execute(
            """
            INSERT INTO runs (
                run_id, preset, n_hospitals, n_rounds, secure_aggregation,
                differential_privacy, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                preset,
                n_hospitals,
                n_rounds,
                int(secure_aggregation),
                int(differential_privacy),
                "running",
                _now(),
            ),
        )
        self._connection.commit()
        return run_id

    def record_round(
        self,
        run_id: str,
        round_index: int,
        accuracy: float,
        log_loss: float | None,
        n_clients: int,
        bytes_exchanged: int,
        duration_s: float,
    ) -> None:
        """
        Store the global metrics for one federated round.

        Parameters
        ----------
        run_id : str
            The run id.
        round_index : int
            1-based round number.
        accuracy : float
            Global accuracy after this round.
        log_loss : float | None
            Global log loss after this round.
        n_clients : int
            Number of participating clients.
        bytes_exchanged : int
            Estimated bytes exchanged during the round.
        duration_s : float
            Wall-clock duration of the round.
        """

        self._connection.execute(
            """
            INSERT INTO rounds (
                run_id, round_index, accuracy, log_loss, n_clients,
                bytes_exchanged, duration_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                round_index,
                accuracy,
                log_loss,
                n_clients,
                bytes_exchanged,
                duration_s,
            ),
        )
        self._connection.commit()

    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed with a completion timestamp."""
        self._connection.execute(
            "UPDATE runs SET status = ?, completed_at = ? WHERE run_id = ?",
            ("completed", _now(), run_id),
        )
        self._connection.commit()

    def register_model(
        self,
        run_id: str,
        preset: str,
        model_path: str | Path,
        accuracy: float | None,
        roc_auc: float | None,
        epsilon: float | None,
    ) -> int:
        """
        Register the global model artifact produced by a run.

        Parameters
        ----------
        run_id : str
            The run id that produced the model.
        preset : str
            Dataset preset the model predicts.
        model_path : str | Path
            Path to the persisted model artifact.
        accuracy : float | None
            Hold-out accuracy of the global model.
        roc_auc : float | None
            Hold-out ROC-AUC of the global model.
        epsilon : float | None
            Worst-case DP epsilon (when differential privacy was active).

        Returns
        -------
        int
            The version number assigned to this model.
        """

        previous = self._connection.execute(
            "SELECT COUNT(*) AS count FROM models WHERE preset = ?", (preset,)
        ).fetchone()
        version = int(previous["count"]) + 1
        self._connection.execute(
            """
            INSERT INTO models (
                run_id, preset, version, model_path, accuracy, roc_auc,
                epsilon, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                preset,
                version,
                str(model_path),
                accuracy,
                roc_auc,
                epsilon,
                _now(),
            ),
        )
        self._connection.commit()
        return version

    def latest_model(self, preset: str | None = None) -> dict[str, Any] | None:
        """
        Return the most recently registered model (optionally per preset).

        Parameters
        ----------
        preset : str | None
            Restrict to a preset when given.

        Returns
        -------
        dict[str, Any] | None
            The latest model row, or ``None`` if none exists.
        """

        query = "SELECT * FROM models"
        params: tuple[Any, ...] = ()
        if preset:
            query += " WHERE preset = ?"
            params = (preset,)
        query += " ORDER BY id DESC LIMIT 1"
        row = self._connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def list_models(self, preset: str | None = None) -> list[dict[str, Any]]:
        """
        List all registered models, newest first.

        Parameters
        ----------
        preset : str | None
            Restrict to a preset when given.

        Returns
        -------
        list[dict[str, Any]]
            Model rows ordered by registration time (descending).
        """

        query = "SELECT * FROM models"
        params: tuple[Any, ...] = ()
        if preset:
            query += " WHERE preset = ?"
            params = (preset,)
        query += " ORDER BY id DESC"
        return [dict(row) for row in self._connection.execute(query, params).fetchall()]

    def list_runs(self, preset: str | None = None) -> list[dict[str, Any]]:
        """
        List all federation runs, newest first.

        Parameters
        ----------
        preset : str | None
            Restrict to a preset when given.

        Returns
        -------
        list[dict[str, Any]]
            Run rows ordered by creation time (descending).
        """

        query = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if preset:
            query += " WHERE preset = ?"
            params = (preset,)
        query += " ORDER BY created_at DESC"
        return [dict(row) for row in self._connection.execute(query, params).fetchall()]

    def run_rounds(self, run_id: str) -> list[dict[str, Any]]:
        """
        Return the per-round metrics of a run in round order.

        Parameters
        ----------
        run_id : str
            The run id.

        Returns
        -------
        list[dict[str, Any]]
            Round rows ordered by round index.
        """

        rows = self._connection.execute(
            "SELECT * FROM rounds WHERE run_id = ? ORDER BY round_index", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


__all__ = ["ModelRegistry"]
