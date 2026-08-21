"""
Persistent store for patient risk history.

Records risk assessments from clinical analyses so longitudinal
trends and escalation alerts can be computed.
"""

from __future__ import annotations

import json
import sqlite3

from datetime import UTC, datetime
from pathlib import Path

from .config import settings
from .schemas import (
    EscalationAlert,
    RiskHistoryRecord,
    RiskHistoryResponse,
    RiskHistorySummary,
    RiskLevel,
    RiskTrend,
)


class RiskHistoryStoreError(Exception):
    """Raised when a risk history store operation fails."""


class RiskHistoryStore:
    """
    SQLite-backed risk history store.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database file. Defaults to ``settings.DB_PATH``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Open (and create) the SQLite database with the risk history table."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    prediction INTEGER,
                    confidence REAL,
                    markers TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Index for efficient patient+preset lookups
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_risk_history_patient_preset
                ON risk_history (patient_id, preset, created_at)
                """
            )
            self._connection.commit()
        return self._connection

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def add(
        self,
        patient_id: str,
        preset: str,
        risk_score: float,
        risk_level: RiskLevel,
        prediction: int | None = None,
        confidence: float | None = None,
        markers: dict[str, float] | None = None,
    ) -> int:
        """
        Persist one risk assessment record.

        Parameters
        ----------
        patient_id : str
            Patient study id.
        preset : str
            Dataset preset.
        risk_score : float
            Risk score in [0, 1].
        risk_level : RiskLevel
            Risk level: "low" / "medium" / "high".
        prediction : int | None
            Predicted class (0/1).
        confidence : float | None
            Model confidence.
        markers : dict[str, float] | None
            Raw clinical markers.

        Returns
        -------
        int
            The new row id.

        Raises
        ------
        RiskHistoryStoreError
            If the row cannot be persisted.
        """
        try:
            cursor = self.connect().execute(
                """
                INSERT INTO risk_history (
                    patient_id, preset, risk_score, risk_level,
                    prediction, confidence, markers, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id,
                    preset,
                    risk_score,
                    risk_level,
                    prediction,
                    confidence,
                    json.dumps(markers) if markers else None,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._connection.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as error:
            raise RiskHistoryStoreError(
                f"Could not persist risk history: {error}"
            ) from error

    def get_patient_history(
        self, patient_id: str, preset: str, limit: int = 100
    ) -> list[RiskHistoryRecord]:
        """
        Return risk history for a patient-preset pair (newest first).

        Parameters
        ----------
        patient_id : str
            Patient study id.
        preset : str
            Dataset preset.
        limit : int
            Maximum number of records to return.

        Returns
        -------
        list[RiskHistoryRecord]
            History records ordered by creation time (descending).
        """
        rows = (
            self.connect()
            .execute(
                """
            SELECT id, patient_id, preset, risk_score, risk_level,
                   prediction, confidence, markers, created_at
            FROM risk_history
            WHERE patient_id = ? AND preset = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
                (patient_id, preset, limit),
            )
            .fetchall()
        )
        return [self._row_to_record(row) for row in rows]

    def get_recent_scores(
        self, patient_id: str, preset: str, limit: int
    ) -> list[tuple[float, RiskLevel, datetime]]:
        """
        Return recent (risk_score, risk_level, created_at) tuples, newest first.
        """
        rows = (
            self.connect()
            .execute(
                """
            SELECT risk_score, risk_level, created_at
            FROM risk_history
            WHERE patient_id = ? AND preset = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
                (patient_id, preset, limit),
            )
            .fetchall()
        )
        return [(r[0], r[1], datetime.fromisoformat(r[2])) for r in rows]

    def get_all_patients(self) -> list[tuple[str, str]]:
        """Return distinct (patient_id, preset) pairs with at least one record."""
        rows = (
            self.connect()
            .execute(
                """
            SELECT DISTINCT patient_id, preset
            FROM risk_history
            ORDER BY patient_id, preset
            """
            )
            .fetchall()
        )
        return [(r[0], r[1]) for r in rows]

    def compute_trend(
        self, patient_id: str, preset: str, window: int | None = None
    ) -> RiskTrend:
        """
        Compute risk trend for a patient-preset over the recent window.

        Parameters
        ----------
        patient_id : str
            Patient study id.
        preset : str
            Dataset preset.
        window : int | None
            Number of recent analyses to consider (default: settings.TREND_WINDOW).

        Returns
        -------
        RiskTrend
            Computed trend with direction, slope, and escalation alert.
        """
        w = window or settings.TREND_WINDOW
        recent = self.get_recent_scores(patient_id, preset, w)
        if len(recent) < settings.MIN_TREND_POINTS:
            return RiskTrend(
                patient_id=patient_id,
                preset=preset,
                recent_scores=[],
                trend_direction="stable",
                slope=0.0,
                avg_score=0.0,
                latest_score=0.0,
                latest_level="low",
                n_points=len(recent),
            )

        scores = [r[0] for r in recent]
        levels = [r[1] for r in recent]
        latest_score = scores[0]
        latest_level = levels[0]

        # Linear regression for slope (older to newer)
        x = list(range(len(scores)))[::-1]  # oldest=0, newest=n-1
        y = scores[::-1]
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.01:
            direction = "worsening"
        elif slope < -0.01:
            direction = "improving"
        else:
            direction = "stable"

        # Escalation alert: check if latest jump exceeds threshold
        escalation = False
        if len(scores) >= 2:
            delta = scores[0] - scores[1]
            if delta > settings.ESCALATION_THRESHOLD:
                escalation = True

        return RiskTrend(
            patient_id=patient_id,
            preset=preset,
            recent_scores=scores,
            trend_direction=direction,
            slope=slope,
            avg_score=sum(scores) / len(scores),
            latest_score=latest_score,
            latest_level=latest_level,
            escalation_alert=escalation,
            n_points=len(scores),
        )

    def get_summary(self, patient_id: str, preset: str) -> RiskHistorySummary:
        """
        Get a summary of risk history for a patient-preset.

        Parameters
        ----------
        patient_id : str
            Patient study id.
        preset : str
            Dataset preset.

        Returns
        -------
        RiskHistorySummary
            Summary with total count, trend, and latest record.
        """
        history = self.get_patient_history(patient_id, preset, limit=1)
        latest = history[0] if history else None
        total = (
            self.connect()
            .execute(
                "SELECT COUNT(*) FROM risk_history WHERE patient_id = ? AND preset = ?",
                (patient_id, preset),
            )
            .fetchone()[0]
        )

        if total >= settings.MIN_TREND_POINTS:
            trend = self.compute_trend(patient_id, preset)
        else:
            trend = None

        return RiskHistorySummary(
            patient_id=patient_id,
            preset=preset,
            total_analyses=total,
            trend=trend,
            latest=latest,
        )

    def get_all_summaries(self) -> RiskHistoryResponse:
        """
        Get summaries for all patient-preset combinations.

        Returns
        -------
        RiskHistoryResponse
            All summaries plus total alert count.
        """
        pairs = self.get_all_patients()
        summaries = []
        alert_count = 0
        for patient_id, preset in pairs:
            summary = self.get_summary(patient_id, preset)
            summaries.append(summary)
            if summary.trend and summary.trend.escalation_alert:
                alert_count += 1
        return RiskHistoryResponse(summaries=summaries, alert_count=alert_count)

    def get_escalation_alerts(self) -> list[EscalationAlert]:
        """
        Return all active escalation alerts.

        An alert is active if the latest score increase exceeds the
        threshold compared to the immediately prior analysis.

        Returns
        -------
        list[EscalationAlert]
            Active alerts sorted by timestamp (newest first).
        """
        alerts = []
        pairs = self.get_all_patients()
        for patient_id, preset in pairs:
            recent = self.get_recent_scores(patient_id, preset, 2)
            if len(recent) >= 2:
                curr_score, _, curr_time = recent[0]
                prev_score, _, _ = recent[1]
                delta = curr_score - prev_score
                if delta > settings.ESCALATION_THRESHOLD:
                    alerts.append(
                        EscalationAlert(
                            patient_id=patient_id,
                            preset=preset,
                            previous_score=prev_score,
                            current_score=curr_score,
                            delta=delta,
                            threshold=settings.ESCALATION_THRESHOLD,
                            timestamp=curr_time,
                        )
                    )
        # Sort by timestamp descending
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts

    @staticmethod
    def _row_to_record(row: sqlite3.Row | tuple) -> RiskHistoryRecord:
        """Convert a SQLite row to a ``RiskHistoryRecord``."""
        return RiskHistoryRecord(
            id=int(row[0]),
            patient_id=row[1],
            preset=row[2],
            risk_score=float(row[3]),
            risk_level=row[4],
            prediction=int(row[5]) if row[5] is not None else None,
            confidence=float(row[6]) if row[6] is not None else None,
            markers=json.loads(row[7]) if row[7] else None,
            created_at=datetime.fromisoformat(row[8]),
        )


__all__ = ["RiskHistoryStore", "RiskHistoryStoreError"]
