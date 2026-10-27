"""
context/profile.py

Per-user accounts, enrolled voiceprints, and transaction history.

WHY THIS EXISTS
Layer 4 asks "is this request normal FOR THIS USER?" -- a question that is
meaningless without a user and without a history to be normal relative to.
Until now the demo had one global enrollment, so L4 could not exist even in
principle. This module supplies the missing state.

HONESTY NOTE THAT MUST APPEAR IN THE PAPER
We have no real deployment, therefore no real usage logs. Transaction
histories here are either (a) accumulated live during a demo session, or
(b) seeded from a declared synthetic generator. Any L4 result computed on
seeded history is a result on SIMULATED behaviour and must be reported as
such. L1 and L3 are evaluated on real corpora; L4 is not, and pretending
otherwise would be the kind of quiet overclaim that invalidates a paper.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "securevoice.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    embedding   TEXT,              -- JSON list, the enrolled ECAPA voiceprint
    n_enroll    INTEGER DEFAULT 0,
    passphrase  TEXT                 -- normalised text of the enrolled phrase
);
CREATE TABLE IF NOT EXISTS transactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    ts         TEXT NOT NULL,
    amount     REAL,
    recipient  TEXT,
    action     TEXT,
    transcript TEXT,
    outcome    TEXT,
    fused_llr  REAL,
    simulated  INTEGER DEFAULT 0,  -- 1 = seeded, not a real observation
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


@dataclass
class UserProfile:
    user_id: str
    display_name: str
    embedding: np.ndarray | None = None
    n_enroll: int = 0
    passphrase: str = ""
    history: list[dict] = field(default_factory=list)

    @property
    def enrolled(self) -> bool:
        return self.embedding is not None

    @property
    def n_real(self) -> int:
        return sum(1 for h in self.history if not h.get("simulated"))

    @property
    def n_simulated(self) -> int:
        return sum(1 for h in self.history if h.get("simulated"))


class ProfileStore:
    """SQLite-backed user store. Small, synchronous, and good enough."""

    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            # Additive migration for databases created before passphrases.
            cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
            if "passphrase" not in cols:
                c.execute("ALTER TABLE users ADD COLUMN passphrase TEXT")

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ users --
    def create_user(self, user_id: str, display_name: str) -> UserProfile:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO users (user_id, display_name, created_at) "
                "VALUES (?, ?, ?)",
                (user_id, display_name, datetime.utcnow().isoformat()),
            )
        return self.get_user(user_id)

    def list_users(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT user_id, display_name, n_enroll FROM users ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user(self, user_id: str) -> UserProfile | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE user_id = ?",
                            (user_id,)).fetchone()
            if row is None:
                return None
            hist = c.execute(
                "SELECT ts, amount, recipient, action, transcript, outcome, simulated "
                "FROM transactions WHERE user_id = ? ORDER BY ts", (user_id,)
            ).fetchall()
        emb = np.array(json.loads(row["embedding"])) if row["embedding"] else None
        return UserProfile(
            user_id=row["user_id"], display_name=row["display_name"],
            embedding=emb, n_enroll=row["n_enroll"],
            passphrase=row["passphrase"] or "",
            history=[dict(h) for h in hist],
        )

    def set_embedding(self, user_id: str, embedding: np.ndarray, n_enroll: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE users SET embedding = ?, n_enroll = ? WHERE user_id = ?",
                      (json.dumps(np.asarray(embedding).tolist()), n_enroll, user_id))

    def set_passphrase(self, user_id: str, normalised_text: str) -> None:
        """
        Store the enrolled challenge phrase.

        Stored as normalised plaintext rather than a hash, because
        verification is fuzzy: ASR will not return the identical string twice,
        so an exact-match hash would reject the legitimate user constantly.
        The consequence is that a database compromise reveals the phrase --
        acceptable here only because the phrase is a SECOND factor layered on
        voice, never a credential on its own. A deployment holding real
        passphrases should use a random per-challenge prompt instead, which
        needs no stored secret at all.
        """
        with self._conn() as c:
            c.execute("UPDATE users SET passphrase = ? WHERE user_id = ?",
                      (normalised_text, user_id))

    def delete_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    # ----------------------------------------------------- transactions --
    def record(self, user_id: str, *, amount: float | None, recipient: str,
               action: str, transcript: str, outcome: str, fused_llr: float,
               simulated: bool = False, ts: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO transactions (user_id, ts, amount, recipient, action, "
                "transcript, outcome, fused_llr, simulated) VALUES (?,?,?,?,?,?,?,?,?)",
                (user_id, ts or datetime.utcnow().isoformat(), amount, recipient,
                 action, transcript, outcome, fused_llr, int(simulated)),
            )

    def seed_history(self, user_id: str, n: int = 40, seed: int = 42) -> int:
        """
        Populate a DECLARED-SYNTHETIC spending history so L4 has something to
        be normal relative to during a demo. Log-normal amounts around a few
        hundred rupees with a small set of familiar recipients -- the shape of
        ordinary consumer spending, not fitted to any real data.
        """
        rng = np.random.default_rng(seed)
        recipients = ["savings", "amazon", "swiggy", "electricity board",
                      "mom", "landlord"]
        rows = []
        for i in range(n):
            amt = float(np.round(rng.lognormal(mean=6.0, sigma=0.9), 0))
            hour = int(np.clip(rng.normal(14, 4), 6, 23))
            ts = datetime(2026, 8, 1 + (i % 28), hour,
                          int(rng.integers(0, 60))).isoformat()
            rows.append((amt, recipients[int(rng.integers(0, len(recipients)))], ts))
        for amt, rcp, ts in rows:
            self.record(user_id, amount=amt, recipient=rcp, action="payment",
                        transcript="", outcome="ALLOW", fused_llr=0.0,
                        simulated=True, ts=ts)
        return len(rows)
