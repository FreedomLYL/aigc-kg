"""云端种子数据：本地导出 / 空库自动导入。

免费托管（如 Render 免费实例）没有持久磁盘，SQLite 会在重启时被清空。
本模块用于：1) 把本机 kg.db 导出为 seed/seed_data.json；2) 服务启动时若库为空，则从种子文件恢复初始账号与图谱，保证演示数据不丢。
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any

from . import config, db

# 课程名不允许带数字后缀（历史遗留 `名称_<时间戳>`），导出/导入一律过滤，杜绝数字后缀课程
_BAD_SUFFIX = re.compile(r"_\d{5,}$")


def _course_ok(cid: str) -> bool:
    return bool(cid) and not _BAD_SUFFIX.search(cid)


# ---------- 导出（在本机执行）----------

def export_seed(db_path: Path, out_file: Path) -> Dict[str, Any]:
    """把指定的 SQLite 库导出为种子 JSON。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        data = {
            "users": [dict(r) for r in conn.execute(
                "SELECT username, password_hash, role, created_at FROM users")],
            "graphs": [dict(r) for r in conn.execute(
                "SELECT course_id, data, updated_at FROM graphs")],
            "files": [dict(r) for r in conn.execute(
                "SELECT username, course_id, filename, category, label, size, text_content, uploaded_at FROM files")],
        }
    finally:
        conn.close()
    # 过滤数字后缀课程及其课程文件
    data["graphs"] = [g for g in data["graphs"] if _course_ok(g["course_id"])]
    ok_courses = {g["course_id"] for g in data["graphs"]}
    data["files"] = [f for f in data["files"] if f["course_id"] in ok_courses]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _seed_payload() -> Dict[str, Any]:
    if not config.SEED_FILE.exists():
        return {"users": [], "graphs": [], "files": []}
    with config.SEED_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- 演示账号兜底 ----------

DEMO_ACCOUNTS = [
    ("teacher", "123456", "teacher"),
    ("student", "123456", "student"),
]


def ensure_demo_users() -> None:
    """保证 teacher/student 演示账号始终存在（无论库为空还是已有旧数据）。"""
    for username, password, role in DEMO_ACCOUNTS:
        if db.user_exists(username):
            continue
        db.create_user(username, password, role)


def apply_seed_if_empty() -> bool:
    """若库为空且启用种子，则把演示数据导入（兼容 SQLite 与 PostgreSQL）。返回是否执行了导入。"""
    if not (config.SEED_ON_START and config.SEED_FILE.exists()):
        return False
    payload = _seed_payload()
    if not payload["users"] and not payload["graphs"] and not payload["files"]:
        return False
    # 一律丢弃数字后缀课程，确保恢复后不会出现 `名称_<时间戳>` 课程
    graphs = [g for g in payload["graphs"] if _course_ok(g["course_id"])]
    ok_courses = {g["course_id"] for g in graphs}
    files = [f for f in payload["files"] if f["course_id"] in ok_courses]
    with db.backend_conn() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        if n > 0:                       # 已有用户，不重复导入
            return False
        for u in payload["users"]:
            c.execute(db.placehint(
                "INSERT INTO users(username, password_hash, role, created_at) "
                "VALUES(%s,%s,%s,%s) ON CONFLICT(username) DO NOTHING"),
                (u["username"], u["password_hash"], u["role"], u.get("created_at", "")))
        for g in graphs:
            c.execute(db.placehint(
                "INSERT INTO graphs(course_id, data, updated_at) VALUES(%s,%s,%s) "
                "ON CONFLICT(course_id) DO UPDATE SET data=excluded.data, "
                "updated_at=excluded.updated_at"),
                (g["course_id"], g["data"], g.get("updated_at", "")))
        for f in files:
            c.execute(db.placehint(
                "INSERT INTO files(username, course_id, filename, category, label, size, "
                "text_content, uploaded_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"),
                (f["username"], f["course_id"], f["filename"], f["category"], f["label"],
                 f["size"], f.get("text_content", ""), f.get("uploaded_at", "")))
    return True


if __name__ == "__main__":
    data = export_seed(config.DB_PATH, config.SEED_FILE)
    print(f"已导出 {config.SEED_FILE}：users={len(data['users'])} "
          f"graphs={len(data['graphs'])} files={len(data['files'])}")