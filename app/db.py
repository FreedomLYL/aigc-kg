"""持久化层：双后端 —— SQLite（本地/默认）或 PostgreSQL（配 DATABASE_URL 自动启用）。

- users   账号表（用户名唯一 + 角色 + 密码哈希）
- sessions 登录会话表（token -> 用户名）
- graphs   课程图谱表（course_id -> 图谱 JSON）
- homework 课后习题布置表（course_id -> 教师 + 布置要求 + 题数）
- quiz_records 学生作答记录表（学情分析的原始数据）

密码用 pbkdf2_hmac 加盐哈希存储，不存明文。
SQL 统一用 `%s` 占位符；SQLite 后端执行前替换为 `?`。时间一律由 Python 传 ISO 字符串，避免方言差异。
"""
import datetime
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
from typing import Dict, Optional
from contextlib import contextmanager

from . import config

_ITERATIONS = 120_000
_lock = threading.Lock()

# 会话表兜底上限：登录本身不设次数限制，仅在会话数超过该阈值时清理最老的一批，
# 避免 sessions 表随登录次数无限膨胀。阈值取较大值，正常使用几乎不会触发。
_MAX_SESSIONS = 10_000

IS_POSTGRES = bool(config.DATABASE_URL)
backend = "postgres" if IS_POSTGRES else "sqlite"


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def placehint(sql: str) -> str:
    """把 SQL 的 %s 占位符换成当前后端所用符号。"""
    return sql.replace("%s", "?") if backend == "sqlite" else sql


if IS_POSTGRES:
    from psycopg.rows import dict_row  # type: ignore


@contextmanager
def backend_conn():
    if IS_POSTGRES:
        import psycopg  # type: ignore
        conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    if IS_POSTGRES:
        ddl = """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            role TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS graphs (
            course_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files (
            id SERIAL PRIMARY KEY, username TEXT NOT NULL, course_id TEXT NOT NULL,
            filename TEXT NOT NULL, category TEXT NOT NULL, label TEXT NOT NULL,
            size BIGINT NOT NULL DEFAULT 0, text_content TEXT NOT NULL DEFAULT '',
            uploaded_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS homework (
            course_id TEXT PRIMARY KEY, teacher_name TEXT NOT NULL,
            requirement TEXT NOT NULL DEFAULT '', n INTEGER NOT NULL DEFAULT 5,
            updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS quiz_records (
            id SERIAL PRIMARY KEY, username TEXT NOT NULL, course_id TEXT NOT NULL,
            total INTEGER NOT NULL DEFAULT 0, correct INTEGER NOT NULL DEFAULT 0,
            wrong_points TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);"""
        with _lock, backend_conn() as c:
            c.execute(ddl)
        return
    with _lock, backend_conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users ("
                  "username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, "
                  "role TEXT NOT NULL, created_at TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS sessions ("
                  "token TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS graphs ("
                  "course_id TEXT PRIMARY KEY, data TEXT NOT NULL, "
                  "updated_at TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS files ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
                  "course_id TEXT NOT NULL, filename TEXT NOT NULL, "
                  "category TEXT NOT NULL, label TEXT NOT NULL, "
                  "size INTEGER NOT NULL DEFAULT 0, text_content TEXT NOT NULL DEFAULT '', "
                  "uploaded_at TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS homework ("
                  "course_id TEXT PRIMARY KEY, teacher_name TEXT NOT NULL, "
                  "requirement TEXT NOT NULL DEFAULT '', n INTEGER NOT NULL DEFAULT 5, "
                  "updated_at TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS quiz_records ("
                  "id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, "
                  "course_id TEXT NOT NULL, total INTEGER NOT NULL DEFAULT 0, "
                  "correct INTEGER NOT NULL DEFAULT 0, "
                  "wrong_points TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL)")


_init_db()


# ---------- 密码 ----------

def _hash(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return salt.hex() + "$" + dk.hex()


def _verify(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split("$")
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return hmac.compare_digest(dk.hex(), dk_hex)


# ---------- 用户 / 会话 ----------

def create_user(username: str, password: str, role: str) -> Dict:
    if role not in ("student", "teacher"):
        raise ValueError("role 必须是 student 或 teacher")
    pwd_hash = _hash(password)
    with _lock, backend_conn() as c:
        c.execute(placehint("INSERT INTO users(username, password_hash, role, created_at) "
                            "VALUES(%s,%s,%s,%s)"),
                  (username, pwd_hash, role, _ts()))
    return {"username": username, "role": role}


def get_user(username: str) -> Optional[Dict]:
    with _lock, backend_conn() as c:
        row = c.execute(placehint(
            "SELECT username, role, password_hash FROM users WHERE username=%s"),
            (username,)).fetchone()
    if not row:
        return None
    return {"username": row["username"], "role": row["role"], "password_hash": row["password_hash"]}


def user_exists(username: str) -> bool:
    return get_user(username) is not None


def login_user(username: str, password: str) -> Optional[Dict]:
    user = get_user(username)
    if not user or not _verify(password, user["password_hash"]):
        return None
    token = uuid.uuid4().hex
    with _lock, backend_conn() as c:
        c.execute(placehint("INSERT INTO sessions(token, username, created_at) VALUES(%s,%s,%s)"),
                  (token, username, _ts()))
        _prune_sessions(c)
    return {"token": token, "username": username, "role": user["role"]}


def _prune_sessions(c, limit: int = _MAX_SESSIONS) -> None:
    """兜底清理：会话数超过上限时删除最老的一批，保证无限次登录但表不无限膨胀。"""
    total = c.execute(placehint("SELECT COUNT(*) AS c FROM sessions")).fetchone()["c"]
    excess = total - limit
    if excess <= 0:
        return
    rows = c.execute(placehint(
        "SELECT token FROM sessions ORDER BY created_at ASC, token ASC LIMIT %s"),
        (excess,)).fetchall()
    ids = [r["token"] for r in rows]
    if not ids:
        return
    marker = "?" if backend == "sqlite" else "%s"
    marks = ",".join(marker for _ in ids)
    c.execute(placehint(f"DELETE FROM sessions WHERE token IN ({marks})"), tuple(ids))


def user_by_token(token: str) -> Optional[Dict]:
    with _lock, backend_conn() as c:
        row = c.execute(
            placehint("SELECT s.token, u.username, u.role FROM sessions s "
                      "JOIN users u ON u.username = s.username WHERE s.token=%s"),
            (token,)).fetchone()
    if not row:
        return None
    return {"token": row["token"], "username": row["username"], "role": row["role"]}


def logout_user(token: str) -> None:
    with _lock, backend_conn() as c:
        c.execute(placehint("DELETE FROM sessions WHERE token=%s"), (token,))


# ---------- 图谱 ----------

def save_graph(course_id: str, data: Dict) -> None:
    with _lock, backend_conn() as c:
        c.execute(placehint("INSERT INTO graphs(course_id, data, updated_at) VALUES(%s,%s,%s) "
                            "ON CONFLICT(course_id) DO UPDATE SET data=excluded.data, "
                            "updated_at=excluded.updated_at"),
                  (course_id, json.dumps(data, ensure_ascii=False), _ts()))


def load_graph(course_id: str) -> Optional[Dict]:
    with _lock, backend_conn() as c:
        row = c.execute(placehint("SELECT data FROM graphs WHERE course_id=%s"),
                        (course_id,)).fetchone()
    if not row:
        return None
    return json.loads(row["data"])


def list_graphs() -> list:
    with _lock, backend_conn() as c:
        rows = c.execute(placehint("SELECT course_id FROM graphs ORDER BY updated_at")).fetchall()
    return [r["course_id"] for r in rows]


# ---------- 课程文件库 ----------

def add_file(username: str, course_id: str, filename: str, category: str,
             label: str, size: int, text_content: str) -> int:
    insert = ("INSERT INTO files(username, course_id, filename, category, label, size, "
              "text_content, uploaded_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)")
    if IS_POSTGRES:
        with _lock, backend_conn() as c:
            row = c.execute(placehint(insert + " RETURNING id"),
                            (username, course_id, filename, category, label, size,
                             text_content, _ts())).fetchone()
            return int(row["id"])
    with _lock, backend_conn() as c:
        cur = c.execute(placehint(insert),
                        (username, course_id, filename, category, label, size,
                         text_content, _ts()))
        return int(cur.lastrowid)


def list_files(username: str, course_id: Optional[str] = None) -> list:
    sql = ("SELECT id, username, course_id, filename, category, label, size, "
           "text_content, uploaded_at FROM files WHERE username=%s")
    params = (username,)
    if course_id:
        sql += " AND course_id=%s"
        params = (username, course_id)
    sql += " ORDER BY uploaded_at DESC"
    with _lock, backend_conn() as c:
        rows = c.execute(placehint(sql), params).fetchall()
    return [{
        "id": r["id"], "course_id": r["course_id"], "filename": r["filename"],
        "category": r["category"], "label": r["label"], "size": r["size"],
        "uploaded_at": r["uploaded_at"],
        "text_len": len(r["text_content"] or ""),
    } for r in rows]


def get_file(file_id: int, username: str) -> Optional[Dict]:
    with _lock, backend_conn() as c:
        row = c.execute(placehint(
            "SELECT id, username, course_id, filename, category, label, text_content "
            "FROM files WHERE id=%s AND username=%s"), (file_id, username)).fetchone()
    if not row:
        return None
    return {
        "id": row["id"], "course_id": row["course_id"], "filename": row["filename"],
        "category": row["category"], "label": row["label"],
        "text_content": row["text_content"],
    }


def delete_file(file_id: int, username: str) -> bool:
    with _lock, backend_conn() as c:
        cur = c.execute(placehint("DELETE FROM files WHERE id=%s AND username=%s"),
                        (file_id, username))
        return (cur.rowcount or 0) > 0


def delete_course(course_id: str) -> None:
    """删除课程图谱及其所属课程文件。"""
    with _lock, backend_conn() as c:
        c.execute(placehint("DELETE FROM graphs WHERE course_id=%s"), (course_id,))
        c.execute(placehint("DELETE FROM files WHERE course_id=%s"), (course_id,))
        c.execute(placehint("DELETE FROM homework WHERE course_id=%s"), (course_id,))
        c.execute(placehint("DELETE FROM quiz_records WHERE course_id=%s"), (course_id,))


# ---------- 课后习题布置（教师） ----------

def save_homework(course_id: str, teacher_name: str, requirement: str, n: int) -> dict:
    """保存/更新某课程的课后习题布置要求（每门课仅保留最新一笔）。"""
    with _lock, backend_conn() as c:
        c.execute(placehint(
            "INSERT INTO homework(course_id, teacher_name, requirement, n, updated_at) "
            "VALUES(%s,%s,%s,%s,%s) ON CONFLICT(course_id) DO UPDATE SET "
            "teacher_name=excluded.teacher_name, requirement=excluded.requirement, "
            "n=excluded.n, updated_at=excluded.updated_at"),
            (course_id, teacher_name, requirement, int(n), _ts()))
    return {
        "course_id": course_id, "teacher_name": teacher_name,
        "requirement": requirement, "n": int(n), "updated_at": _ts(),
    }


def clear_homework(course_id: str) -> None:
    with _lock, backend_conn() as c:
        c.execute(placehint("DELETE FROM homework WHERE course_id=%s"), (course_id,))


def get_homework(course_id: str) -> Optional[Dict]:
    with _lock, backend_conn() as c:
        row = c.execute(placehint(
            "SELECT course_id, teacher_name, requirement, n, updated_at "
            "FROM homework WHERE course_id=%s"), (course_id,)).fetchone()
    if not row:
        return None
    return {
        "course_id": row["course_id"], "teacher_name": row["teacher_name"],
        "requirement": row["requirement"], "n": row["n"], "updated_at": row["updated_at"],
    }


# ---------- 学生作答记录（学情分析数据源） ----------

def add_quiz_record(username: str, course_id: str, total: int, correct: int,
                    wrong_points: list) -> int:
    insert = ("INSERT INTO quiz_records(username, course_id, total, correct, "
              "wrong_points, created_at) VALUES(%s,%s,%s,%s,%s,%s)")
    if IS_POSTGRES:
        with _lock, backend_conn() as c:
            row = c.execute(placehint(insert + " RETURNING id"),
                            (username, course_id, int(total), int(correct),
                             json.dumps(wrong_points, ensure_ascii=False), _ts())).fetchone()
            return int(row["id"])
    with _lock, backend_conn() as c:
        cur = c.execute(placehint(insert),
                        (username, course_id, int(total), int(correct),
                         json.dumps(wrong_points, ensure_ascii=False), _ts()))
        return int(cur.lastrowid)


def list_quiz_records(course_id: str) -> list:
    with _lock, backend_conn() as c:
        rows = c.execute(placehint(
            "SELECT username, total, correct, wrong_points, created_at "
            "FROM quiz_records WHERE course_id=%s ORDER BY created_at"),
            (course_id,)).fetchall()
    out = []
    for r in rows:
        try:
            wp = json.loads(r["wrong_points"] or "[]")
        except Exception:
            wp = []
        out.append({
            "username": r["username"], "total": r["total"], "correct": r["correct"],
            "wrong_points": wp, "created_at": r["created_at"],
        })
    return out


def list_all_quiz_courses() -> list:
    with _lock, backend_conn() as c:
        rows = c.execute(
            "SELECT DISTINCT course_id FROM quiz_records ORDER BY course_id").fetchall()
    return [r["course_id"] for r in rows]