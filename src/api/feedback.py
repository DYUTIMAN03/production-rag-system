"""
Feedback Store — SQLite-backed user feedback collection.
Tracks thumbs-up/down on RAG responses for continuous improvement.
"""

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FeedbackEntry:
    """A single feedback record."""
    id: int
    timestamp: float
    question: str
    answer: str
    citations_json: str
    is_positive: bool
    comment: str = ""


class FeedbackStore:
    """
    SQLite-backed feedback storage.

    Captures user satisfaction signals (thumbs up/down) linked to
    the query, response, and citations. This enables:
    - Identifying retrieval failures (negative feedback patterns)
    - Improving the golden dataset with real user queries
    - Measuring user satisfaction over time
    """

    def __init__(self, db_path: str = "./feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the feedback table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations_json TEXT DEFAULT '[]',
                is_positive INTEGER NOT NULL,
                comment TEXT DEFAULT '',
                confidence_score REAL DEFAULT 0.0,
                chunks_used INTEGER DEFAULT 0,
                is_grounded INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()

    def save_feedback(
        self,
        question: str,
        answer: str,
        is_positive: bool,
        citations: list = None,
        comment: str = "",
        confidence_score: float = 0.0,
        chunks_used: int = 0,
        is_grounded: bool = True,
    ) -> int:
        """
        Save user feedback for a query-response pair.

        Args:
            question: The user's original question
            answer: The generated answer
            is_positive: True for thumbs-up, False for thumbs-down
            citations: List of citation dicts
            comment: Optional text feedback
            confidence_score: Confidence score from the pipeline
            chunks_used: Number of chunks used in the answer
            is_grounded: Whether the answer was grounded

        Returns:
            The ID of the saved feedback entry
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO feedback
               (timestamp, question, answer, citations_json, is_positive,
                comment, confidence_score, chunks_used, is_grounded)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                question,
                answer,
                json.dumps(citations or []),
                1 if is_positive else 0,
                comment,
                confidence_score,
                chunks_used,
                1 if is_grounded else 0,
            ),
        )
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return feedback_id

    def get_summary(self) -> dict:
        """Get aggregated feedback statistics."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE is_positive = 1"
        ).fetchone()[0]
        negative = total - positive

        # Recent feedback (last 24h)
        day_ago = time.time() - 86400
        recent_total = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE timestamp > ?", (day_ago,)
        ).fetchone()[0]
        recent_positive = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE is_positive = 1 AND timestamp > ?",
            (day_ago,),
        ).fetchone()[0]

        conn.close()

        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / total, 4) if total > 0 else 0,
            "recent_24h": {
                "total": recent_total,
                "positive": recent_positive,
                "negative": recent_total - recent_positive,
                "satisfaction_rate": round(
                    recent_positive / recent_total, 4
                ) if recent_total > 0 else 0,
            },
        }

    def get_negative_feedback(self, limit: int = 20) -> List[dict]:
        """
        Get recent negative feedback entries for analysis.
        These represent queries where the system failed users.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT id, timestamp, question, answer, comment, confidence_score
               FROM feedback WHERE is_positive = 0
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "question": r[2],
                "answer": r[3][:200] + "..." if len(r[3]) > 200 else r[3],
                "comment": r[4],
                "confidence_score": r[5],
            }
            for r in rows
        ]
