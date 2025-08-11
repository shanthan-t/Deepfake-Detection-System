import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
from pathlib import Path
from priva_main.config import DATA_DIR

DB_PATH = Path(DATA_DIR) / "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login TEXT
    )''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS detection_history (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        file_name TEXT NOT NULL,
        file_type TEXT NOT NULL,
        is_fake INTEGER NOT NULL,
        confidence REAL NOT NULL,
        detection_time TEXT NOT NULL
    )''')
    conn.commit(); conn.close()

def hash_password(password, salt):
    import hashlib
    return hashlib.sha256((password + salt).encode()).hexdigest()

class User:
    @staticmethod
    def create(username, email, password, salt):
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, hash_password(password, salt), datetime.now().isoformat())
            )
            conn.commit(); return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    @staticmethod
    def authenticate(username, password, salt):
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, hash_password(password, salt))
        ).fetchone()
        if user:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?",
                         (datetime.now().isoformat(), user['id']))
            conn.commit()
        conn.close()
        return user

class DetectionHistory:
    @staticmethod
    def add_entry(user_id, file_name, file_type, is_fake, confidence):
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO detection_history (user_id, file_name, file_type, is_fake, confidence, detection_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, file_name, file_type, int(is_fake), float(confidence), datetime.now().isoformat())
        )
        conn.commit(); conn.close()

    @staticmethod
    def get_user_history(user_id):
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT * FROM detection_history WHERE user_id = ? ORDER BY detection_time DESC",
            (user_id,)
        ).fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

init_db()
