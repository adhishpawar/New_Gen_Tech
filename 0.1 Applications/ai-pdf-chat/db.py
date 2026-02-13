import sqlite3
from datetime import datetime
from typing import List, Dict

DB_PATH = "metadata.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS pdfs (
    id TEXT PRIMARY KEY,
    filename TEXT,
    path TEXT,
    created_at TEXT
)
""")
conn.commit()

def add_pdf(pdf_id: str, filename: str, path: str):
    cur.execute("INSERT INTO pdfs (id, filename, path, created_at) VALUES (?, ?, ?, ?)",
                (pdf_id, filename, path, datetime.utcnow().isoformat()))
    conn.commit()

def list_pdfs() -> List[Dict]:
    cur.execute("SELECT id, filename, path, created_at FROM pdfs ORDER BY created_at DESC")
    rows = cur.fetchall()
    return [{"id": r[0], "filename": r[1], "path": r[2], "created_at": r[3]} for r in rows]

def get_pdf(pdf_id: str):
    cur.execute("SELECT id, filename, path, created_at FROM pdfs WHERE id = ?", (pdf_id,))
    r = cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "filename": r[1], "path": r[2], "created_at": r[3]}
