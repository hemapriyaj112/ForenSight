"""
database/db.py — SQLite persistence layer for ForenSight analysis results.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.types import AnalysisResult

log = get_logger("forensight.db")

_DEFAULT_DB = "forensight.db"


class Database:
    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id            TEXT PRIMARY KEY,
                run_id                 TEXT,
                video_path             TEXT,
                verdict                TEXT,
                fused_score            REAL,
                video_score            REAL,
                audio_score            REAL,
                calibrated_video_score REAL,
                calibrated_audio_score REAL,
                metadata_json          TEXT,
                created_at             TEXT
            )
        """)
        conn.commit()

    def save_result(self, result: AnalysisResult) -> None:
        meta = result.metadata
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO analyses (
                analysis_id, run_id, video_path, verdict, fused_score,
                video_score, audio_score,
                calibrated_video_score, calibrated_audio_score,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.analysis_id,          # compat property → metadata["run_id"]
            meta.get("run_id"),
            meta.get("video_path"),
            result.verdict.value,
            result.fused_score,
            meta.get("video_score"),
            meta.get("audio_score"),
            meta.get("calibrated_video_score"),
            meta.get("calibrated_audio_score"),
            json.dumps(meta),
            datetime.now(tz=timezone.utc).isoformat(),
        ))
        conn.commit()
        log.info("Saved analysis %s (verdict=%s)",
                 result.analysis_id, result.verdict.value)

    def get_result(self, analysis_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        row  = conn.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


_default_db: Database | None = None


def _get_db() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database()
    return _default_db


def save_result(result: AnalysisResult) -> None:
    _get_db().save_result(result)