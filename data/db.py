# data/db.py
import sqlite3, os, datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "scamguard.db")
os.makedirs(DATA_DIR, exist_ok=True)


def save_review_answers(
    log_id: int,
    user_id: str,
    source: str | None,
    money_asked: str | None,
    money_details: str | None,
    personal_info_asked: str | None,
    personal_info_details: str | None,
    extra_notes: str | None,
):
    """Insert one row into review_answers for this prediction log."""
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO review_answers (
                log_id,
                user_id,
                source,
                money_asked,
                money_details,
                personal_info_asked,
                personal_info_details,
                extra_notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(log_id),
                str(user_id),
                (source or "").strip(),
                (money_asked or "").strip(),
                (money_details or "").strip(),
                (personal_info_asked or "").strip(),
                (personal_info_details or "").strip(),
                (extra_notes or "").strip(),
                ts,
            ),
        )
        conn.commit()


def fetch_review_answers_for_logs(log_ids: list[int]) -> dict[int, dict]:
    """Return a dict: {log_id: {source:..., money_asked:..., ...}}."""
    if not log_ids:
        return {}
    placeholders = ",".join("?" for _ in log_ids)
    with get_db() as conn:
        cur = conn.execute(
            f"""
            SELECT
                log_id,
                source,
                money_asked,
                money_details,
                personal_info_asked,
                personal_info_details,
                extra_notes
            FROM review_answers
            WHERE log_id IN ({placeholders})
            ORDER BY id DESC
            """,
            list(log_ids),
        )
        rows = cur.fetchall()

    answers_by_log: dict[int, dict] = {}
    for r in rows:
        lid = int(r["log_id"])
        # keep the latest row only (ORDER BY id DESC above)
        if lid not in answers_by_log:
            answers_by_log[lid] = {
                "source": r["source"] or "",
                "money_asked": r["money_asked"] or "",
                "money_details": r["money_details"] or "",
                "personal_info_asked": r["personal_info_asked"] or "",
                "personal_info_details": r["personal_info_details"] or "",
                "extra_notes": r["extra_notes"] or "",
            }
    return answers_by_log

# data/db.py

def get_admin_stats() -> dict:
    """
    Return a few high-level stats for the admin dashboard.
    - total_scans: total rows in prediction_logs
    - needs_review: how many are currently in 'Needs Review'
    - avg_confidence: average confidence % over all logs that have confidence
    """
    with get_db() as conn:
        # total logs
        cur = conn.execute("SELECT COUNT(*) AS c FROM prediction_logs")
        total_scans = cur.fetchone()["c"] or 0

        # how many are currently Needs Review
        cur = conn.execute(
            "SELECT COUNT(*) AS c FROM prediction_logs WHERE prediction_label = 'Needs Review'"
        )
        needs_review = cur.fetchone()["c"] or 0

        # average confidence over logs that have a non-null confidence
        cur = conn.execute(
            "SELECT AVG(confidence) AS avg_c FROM prediction_logs WHERE confidence IS NOT NULL"
        )
        row = cur.fetchone()
        avg_confidence = row["avg_c"] if row and row["avg_c"] is not None else 0.0

    return {
        "total_scans": int(total_scans),
        "needs_review": int(needs_review),
        "avg_confidence": round(float(avg_confidence), 2),
    }

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_column(conn, name: str, type_sql: str):
    """Best-effort: add column if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE prediction_logs ADD COLUMN {name} {type_sql}")
    except sqlite3.OperationalError:
        # column already exists or table missing (created below)
        pass

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            text_excerpt TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL,
            was_needs_review INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS review_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            source TEXT,
            money_asked TEXT,
            money_details TEXT,
            personal_info_asked TEXT,
            personal_info_details TEXT,
            extra_notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (log_id) REFERENCES prediction_logs(id)
        )
        """)

        conn.commit()

MAX_EXCERPT_LEN = None  # now we want full text; set a number if you later want to cap

def insert_log(
    user_id: str,
    text_excerpt: str,
    prediction_label: str,
    confidence: float | None,
    was_needs_review: int = 0
):
    if not text_excerpt:
        text_excerpt = "(empty)"
    text_excerpt = text_excerpt.strip().replace("\x00", "")

    ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO prediction_logs
                (user_id, text_excerpt, prediction_label, confidence, was_needs_review, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(user_id), text_excerpt, prediction_label, confidence, int(was_needs_review), ts),
        )
        conn.commit()
        return cur.lastrowid


def save_review_context(
    log_id: int,
    source: str,
    pay_fee: str,
    personal_info: str,
    notes: str
):
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET review_source = ?,
                review_pay_fee = ?,
                review_personal_info = ?,
                review_notes = ?
            WHERE id = ?
        """, (source, pay_fee, personal_info, notes, int(log_id)))
        conn.commit()

def fetch_logs(user_id: str, limit: int = 200):
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id,
                   text_excerpt,
                   prediction_label,
                   confidence,
                   was_needs_review,
                   created_at
            FROM prediction_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (str(user_id), int(limit)))
        return cur.fetchall()

def delete_log(user_id: str, log_id: int) -> int:
    """Delete a single log that belongs to this user. Returns rows affected (0 or 1)."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM prediction_logs WHERE id = ? AND user_id = ?",
            (int(log_id), str(user_id)),
        )
        conn.commit()
        return cur.rowcount

def fetch_review_logs(limit: int = 300):
    """Return logs marked as 'Needs Review' across ALL users."""
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            WHERE LOWER(prediction_label) = 'needs review'
            ORDER BY created_at DESC
            LIMIT ?
        """, (int(limit),))
        return cur.fetchall()

def fetch_all_logs(limit: int = 500):
    """Return logs for admin view across ALL users."""
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            ORDER BY created_at DESC
            LIMIT ?
        """, (int(limit),))
        return cur.fetchall()

def update_log_label(log_id: int, new_label: str):
    """Allow admin to correct a label."""
    new_label = new_label.strip().title()
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET prediction_label = ?
            WHERE id = ?
        """, (new_label, int(log_id)))
        conn.commit()

def fetch_logs_by_label(label: str, limit: int = 300):
    """Return logs filtered by Real/Fake/Needs Review."""
    label = label.strip().lower()
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, user_id, text_excerpt, prediction_label, confidence, created_at
            FROM prediction_logs
            WHERE LOWER(prediction_label) = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (label, int(limit)))
        return cur.fetchall()

def count_needs_review(user_id: str) -> int:
    with get_db() as conn:
        cur = conn.execute("""
            SELECT COUNT(*)
            FROM prediction_logs
            WHERE user_id = ? AND prediction_label = 'Needs Review'
        """, (str(user_id),))
        row = cur.fetchone()
        return int(row[0] if row else 0)

def mark_review_decision(log_id: int, final_label: str) -> None:
    """
    Admin decided on a Needs Review item.
    We set prediction_label to 'Real'/'Fake' BUT keep was_needs_review = 1.
    """
    with get_db() as conn:
        conn.execute("""
            UPDATE prediction_logs
            SET prediction_label = ?, was_needs_review = 1
            WHERE id = ?
        """, (final_label, int(log_id)))
        conn.commit()