"""集中配置。优先读环境变量，其次读 .env 文件。"""
import os
from pathlib import Path
from dotenv import load_dotenv

# app/config.py 的上一级即项目根
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---- 大模型 ----
# LLM_PROVIDER: deepseek（默认）| doubao（火山方舟，OpenAI 兼容），二者都支持真实三元组抽取
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "").strip()
DOUBAO_BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").strip()
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-250815").strip()

# ---- 图数据库 ----
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

# ---- SQLite（默认持久化存储：用户/会话/图谱 一套数据库）----
DB_PATH = Path(os.getenv("KG_DB_PATH", str(_PROJECT_ROOT / "kg.db")))

# ---- 云端 PostgreSQL（可选）：配置 DATABASE_URL 后自动切换为 Postgres 持久化 ----
# 例如：postgresql://user:password@host:5432/dbname?sslmode=require
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# ---- 云端种子：无持久磁盘的免费托管（如 Render）上，服务启动时把内置演示数据灌入空库 ----
# SEED_ON_START=1 时，若库为空则从 seed/seed_data.json 恢复初始数据
SEED_ON_START = os.getenv("SEED_ON_START", "0").strip() == "1"
SEED_FILE = Path(os.getenv("SEED_FILE", str(_PROJECT_ROOT / "seed" / "seed_data.json")))


# 存储层自动判断：有 Neo4j 配置则用图库，否则 SQLite 持久化
def use_neo4j() -> bool:
    return bool(NEO4J_URI and NEO4J_PASSWORD)


# 是否启用模拟抽取：未配置所选 provider 的有效 API Key 时为 True
def use_mock_llm() -> bool:
    if LLM_PROVIDER == "doubao":
        return not DOUBAO_API_KEY
    return not LLM_API_KEY


# 前端静态目录
STATIC_DIR = _PROJECT_ROOT / "static"