"""FastAPI 后端入口：上传解析 → LLM 抽取 → 建图 → 图谱/问答/推荐/编辑接口 + 登录鉴权。"""
import re
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List

from . import document_parser, llm_extractor, graph_store, pathfinder, rag_chat, db, assistant, classify, seed
from .schemas import (
    RequestChat, RecommendRequest,
    NodeCreate, NodeUpdate, EdgeCreate, EdgeDelete,
    RegisterRequest, LoginRequest, AssistantRequest, CourseCreate, ExtractionResult,
)
from .config import STATIC_DIR

app = FastAPI(title="AIGC 课程知识图谱", version="0.5.0")

# 单文件前端（index.html 全部内联，无需静态目录服务）
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")

# 静态资源（本地 ECharts 等），保证离线便携环境无需联网也能渲染图谱
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

extractor = llm_extractor.build_extractor()
chat_engine = rag_chat.build_chat_engine()
teacher_assistant = assistant.build_assistant()

# 免费托管无持久磁盘：空库时从种子恢复演示数据（SEED_ON_START=1 且库为空才生效）
seed.apply_seed_if_empty()
# 兜底：无论库处于何种状态，都确保 teacher/student 演示账号可用（一键快速登录依赖它们）
seed.ensure_demo_users()


# ---------- 认证依赖 ----------

def _token_of(authorization: str):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):].strip()


def require_user(authorization: str = Header("")):
    token = _token_of(authorization)
    user = db.user_by_token(token) if token else None
    if not user:
        raise HTTPException(401, "未登录或登录已失效")
    return user


def require_teacher(authorization: str = Header("")) -> dict:
    user = require_user(authorization)
    if user["role"] != "teacher":
        raise HTTPException(403, "仅教师可进行此操作")
    return user


# ---------- 认证 ----------

@app.post("/api/auth/register")
def register(body: RegisterRequest):
    if db.user_exists(body.username):
        raise HTTPException(409, "用户名已存在")
    if body.role not in ("student", "teacher"):
        raise HTTPException(400, "角色只能是 student 或 teacher")
    db.create_user(body.username, body.password, body.role)
    session = db.login_user(body.username, body.password)
    session["created"] = True
    return session


@app.post("/api/auth/login")
def login(body: LoginRequest):
    session = db.login_user(body.username, body.password)
    if not session:
        raise HTTPException(401, "用户名或密码错误")
    return session


@app.post("/api/auth/logout")
def logout(authorization: str = Header("")):
    token = _token_of(authorization)
    if token:
        db.logout_user(token)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(require_user)):
    return {"username": user["username"], "role": user["role"]}


# ---------- 课程与图谱查询 ----------

@app.get("/api/courses")
def courses(_: dict = Depends(require_user)):
    return {"courses": graph_store.get_store().list_graphs()}


@app.post("/api/courses")
def create_course(body: CourseCreate, _: dict = Depends(require_teacher)):
    """教师创建一门新课程（先生成空图谱，便于后续上传文件自动构图）。"""
    course_id = body.course_id.strip()
    if not course_id:
        raise HTTPException(400, "课程名称不能为空")
    if any(ch.isdigit() for ch in course_id):
        raise HTTPException(400, "课程名称不能包含数字")
    store = graph_store.get_store()
    if store.get_graph(course_id):
        raise HTTPException(409, f"课程已存在：{course_id}")
    store.save(course_id, ExtractionResult())
    return {"course_id": course_id, "graph": store.get_graph(course_id)}


@app.get("/api/graph/{course_id}")
def get_graph(course_id: str, _: dict = Depends(require_user)):
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    return graph


@app.delete("/api/graph/{course_id}")
def delete_course(course_id: str, _: dict = Depends(require_teacher)):
    graph_store.get_store().delete_course(course_id)
    return {"ok": True}


# ---------- 知识抽取（教师） ----------

@app.post("/api/knowledge/extract")
async def extract(course_id: str, file: UploadFile = File(...), _: dict = Depends(require_teacher)):
    raw = await file.read()
    text = document_parser.extract_text_from_bytes(raw, file.filename or "upload.txt")
    result = extractor.extract(text)
    graph_store.get_store().save(course_id, result)
    # 抽取后刷新问答引擎的知识索引
    chat_engine.index_course(course_id, result)
    return {
        "course_id": course_id,
        "extractor": getattr(extractor, "name", "deepseek"),
        "entities": len(result.entities),
        "relations": len(result.relations),
        "graph": graph_store.get_store().get_graph(course_id),
    }


# 模板课程 → 真实课程讲义（samples/ 下的真实课程资料，作为模板课程的默认知识库）。
# 未上传文件时，模板课程自动构图从这些真实讲义抽取，而非预置手写关键词串。
_SUBJECT_FILES = {
    "导论": "machine_learning.md",
    "线性回归": "linear_regression.md",
    "分类算法": "classification_algorithms.md",
    "聚类与降维": "clustering_and_dim_reduction.md",
    "模型评估": "model_evaluation.md",
    "神经网络基础": "nn_basics.md",
    "卷积神经网络": "cnn.md",
    "序列建模": "sequence_modeling.md",
    "Transformer与注意力机制": "transformer_attention.md",
}
# 根课程（机器学习 / 深度学习）的总览讲义
_ROOT_FILES = {"机器学习": "machine_learning.md", "深度学习": "deep_learning.md"}
_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _read_sample(filename: str) -> str:
    try:
        return (_SAMPLES_DIR / filename).read_text(encoding="utf-8")
    except OSError:
        return ""


def default_course_text(course_id: str) -> str:
    """依据课程名返回对应的真实课程讲义，作为模板默认知识库，供未上传文件时自动构图。"""
    seg = course_id.split("-", 1)[-1].strip() if "-" in course_id else course_id
    fname = _SUBJECT_FILES.get(seg) or _ROOT_FILES.get(course_id) or _ROOT_FILES.get(seg)
    if fname:
        text = _read_sample(fname)
        if text:
            return text
    # 兜底：无对应讲义时退回机器学习总览，仍为真实课程资料
    fallback = _read_sample(_ROOT_FILES["机器学习"])
    return fallback if fallback else (
        f"{course_id}\n机器学习 监督学习 无监督学习 强化学习 特征 标签 "
        "训练集 测试集 过拟合 泛化能力 损失函数")


@app.post("/api/graph/{course_id}/auto")
def auto_generate_graph(course_id: str, _: dict = Depends(require_user)):
    """选中模板课程但尚未上传文件时，按课程名生成默认知识图谱。"""
    store = graph_store.get_store()
    if store.get_graph(course_id):
        return {"graph": store.get_graph(course_id), "generated": False}
    if any(ch.isdigit() for ch in course_id):
        raise HTTPException(400, "课程名称不能包含数字")
    result = extractor.extract(default_course_text(course_id))
    store.save(course_id, result)
    chat_engine.index_course(course_id, result)
    return {
        "graph": store.get_graph(course_id),
        "generated": True,
        "extractor": getattr(extractor, "name", "deepseek"),
        "entities": len(result.entities),
        "relations": len(result.relations),
    }


def _sanitize_course_name(raw: str) -> str:
    """由上传文件名派生新课程名：去掉扩展名、数字与多余空白（课程名不能含数字）。"""
    stem = re.sub(r"\.(md|txt|docx?|pdf|pptx?|csv|markdown)$", "", raw or "", flags=re.IGNORECASE)
    name = re.sub(r"\d+", "", stem).strip().replace(" ", "-")
    name = re.sub(r"-+", "-", name)
    return name or "新课程"


# 判定上传文件是否“相关/融合”的知识点重合阈值
_MERGE_THRESHOLD = 3


@app.post("/api/knowledge/build")
async def knowledge_build(course: str = "", file: UploadFile = File(...),
                          user: dict = Depends(require_user)):
    """智能构建图谱：按下拉选择与否 + 上传文件与课程关联与否，四种情况自动决策。

    - 未选课程 + 文件无关 → 新建课程，直接生成上传文件图谱
    - 未选课程 + 文件关联某已有课程 → 融合该课程已有文件与上传文件
    - 已选课程 + 文件无关 → 直接生成上传文件图谱
    - 已选课程 + 文件关联 → 融合所选课程已有文件与上传文件
    """
    raw = await file.read()
    text = document_parser.extract_text_from_bytes(raw, file.filename or "upload.txt")
    result = extractor.extract(text)
    ent_names = [e.name for e in result.entities if e.name]

    def _texts_of(cid: str) -> str:
        parts = []
        for f in db.list_files(user["username"], cid):
            g = db.get_file(f["id"], user["username"])
            if g and g.get("text_content"):
                parts.append(g["text_content"])
        return "\n".join(parts)

    course = (course or "").strip()
    target, created = "", False
    if not course:  # 未在下拉中选择
        best, best_score = "", 0
        for cid in graph_store.get_store().list_graphs():
            t = _texts_of(cid)
            if not t:
                continue
            score = sum(1 for e in ent_names if e and e in t)
            if score and score > best_score:
                best, best_score = cid, score
        if best and best_score >= _MERGE_THRESHOLD:  # 情况二：匹配到已有课程 → 融合
            target = best
        else:                              # 情况一：新建课程，直接生成
            target = _sanitize_course_name(file.filename or "")
            created = True
    else:                                 # 情况三/四：使用所选课程
        target = course

    existing_text = _texts_of(target)
    ent_in_existing = [e for e in ent_names if e and existing_text and e in existing_text]
    merge = len(ent_in_existing) >= _MERGE_THRESHOLD   # 关联→融合；否则直接生成
    final = extractor.extract(existing_text + "\n" + text) if merge else result

    klass = classify.classify(text)
    db.add_file(user["username"], target, file.filename or "未命名.txt",
                klass["category"], klass["label"], len(raw), text)
    graph_store.get_store().save(target, final)
    chat_engine.index_course(target, final)
    return {
        "course_id": target,
        "created": created,
        "merged": merge,
        "entities": len(final.entities),
        "relations": len(final.relations),
        "extractor": getattr(extractor, "name", "deepseek"),
        "graph": graph_store.get_store().get_graph(target),
    }


# ---------- 图谱手动编辑（教师） ----------

@app.post("/api/graph/{course_id}/node")
def add_node(course_id: str, body: NodeCreate, _: dict = Depends(require_teacher)):
    store = graph_store.get_store()
    if not store.get_graph(course_id):
        raise HTTPException(404, f"未找到课程：{course_id}")
    ok = store.add_node(course_id, body.name, body.kind, body.definition or "")
    if not ok:
        raise HTTPException(409, f"节点已存在或添加失败：{body.name}")
    return {"graph": store.get_graph(course_id)}


@app.put("/api/graph/{course_id}/node/{node_name}")
def update_node(course_id: str, node_name: str, body: NodeUpdate, _: dict = Depends(require_teacher)):
    store = graph_store.get_store()
    if not store.get_graph(course_id):
        raise HTTPException(404, f"未找到课程：{course_id}")
    ok = store.update_node(course_id, node_name, body.name, body.kind, body.definition)
    if not ok:
        raise HTTPException(404, f"节点不存在：{node_name}")
    return {"graph": store.get_graph(course_id)}


@app.delete("/api/graph/{course_id}/node/{node_name}")
def delete_node(course_id: str, node_name: str, _: dict = Depends(require_teacher)):
    store = graph_store.get_store()
    if not store.get_graph(course_id):
        raise HTTPException(404, f"未找到课程：{course_id}")
    ok = store.delete_node(course_id, node_name)
    if not ok:
        raise HTTPException(404, f"节点不存在：{node_name}")
    return {"graph": store.get_graph(course_id)}


@app.post("/api/graph/{course_id}/edge")
def add_edge(course_id: str, body: EdgeCreate, _: dict = Depends(require_teacher)):
    store = graph_store.get_store()
    if not store.get_graph(course_id):
        raise HTTPException(404, f"未找到课程：{course_id}")
    ok = store.add_edge(course_id, body.source, body.target, body.type)
    if not ok:
        raise HTTPException(409, "关系已存在或端点节点不存在")
    return {"graph": store.get_graph(course_id)}


@app.delete("/api/graph/{course_id}/edge")
def delete_edge(course_id: str, body: EdgeDelete, _: dict = Depends(require_teacher)):
    store = graph_store.get_store()
    if not store.get_graph(course_id):
        raise HTTPException(404, f"未找到课程：{course_id}")
    ok = store.delete_edge(course_id, body.source, body.target, body.type)
    if not ok:
        raise HTTPException(404, "关系不存在")
    return {"graph": store.get_graph(course_id)}


# ---------- 学习路径推荐 ----------

@app.post("/api/recommend/{course_id}")
def recommend(course_id: str, body: RecommendRequest, _: dict = Depends(require_user)):
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    return {"recommendations": pathfinder.recommend(graph, body.mastered, body.weak, body.top_k)}


# ---------- 智能问答（RAG 升级） ----------

@app.post("/api/chat/{course_id}")
def chat(course_id: str, body: RequestChat, _: dict = Depends(require_user)):
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    history = [{"role": m.role, "content": m.content} for m in body.history]
    return chat_engine.ask(course_id, body.question, graph, history, provider=body.provider)


@app.get("/api/homework")
def homework_get(course_id: str, n: int = 5, requirement: str = "", _: dict = Depends(require_user)):
    """课后习题：学生取题（自动携带教师布置要求）；教师生成预览（requirement 参数不保存）。

    - 若该课程已有教师布置的要求，则优先按布置要求出题；
    - requirement 参数用于教师「生成预览」，仅本次生效、不入库；
    - n<=0 时不生成题目，仅返回布置状态（供教师面板展示）。
    """
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    from . import db as db_
    from . import quiz as quiz_engine
    hw = db_.get_homework(course_id)
    req = requirement.strip() or (hw.get("requirement") if hw else "") or ""
    result = {"assigned": bool(hw), "course_id": course_id,
              "requirement": hw.get("requirement", "") if hw else "",
              "teacher_name": hw.get("teacher_name", "") if hw else "",
              "updated_at": hw.get("updated_at", "") if hw else "",
              "count": hw.get("count", n) if hw else n}
    if int(n or 0) > 0:
        generated = quiz_engine.generate_quiz(graph, int(n), req)
        result.update(generated)
    return result


@app.post("/api/homework/assign")
def homework_assign(body: dict, user: dict = Depends(require_user)):
    """教师布置课后习题：保存布置要求与题数，学生端立即可见。"""
    if user["role"] != "teacher":
        raise HTTPException(403, "只有教师可以布置课后习题")
    course_id = str(body.get("course_id") or "").strip()
    if not course_id:
        raise HTTPException(400, "缺少课程")
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    from . import db as db_
    n = max(1, min(int(body.get("n") or 5), 50))
    return db_.upsert_homework(course_id, str(body.get("requirement") or "").strip(), n, user["username"])


@app.post("/api/homework/clear")
def homework_clear(body: dict, user: dict = Depends(require_user)):
    """教师撤销某课程的课后习题布置。"""
    if user["role"] != "teacher":
        raise HTTPException(403, "只有教师可以撤销课后习题")
    course_id = str(body.get("course_id") or "").strip()
    if not course_id:
        raise HTTPException(400, "缺少课程")
    from . import db as db_
    db_.clear_homework(course_id)
    return {"ok": True, "course_id": course_id}


@app.post("/api/quiz/submit")
def quiz_submit(body: dict, user: dict = Depends(require_user)):
    """保存一次课后习题作答记录（含答错的知识点），用于学习报告与教师学情。"""
    course_id = str(body.get("course_id") or "").strip()
    if not course_id:
        raise HTTPException(400, "缺少课程")
    from . import db as db_
    wrong = [w for w in (body.get("wrong_points") or []) if str(w or "").strip()]
    try:
        total = max(0, int(body.get("total") or 0))
        correct = max(0, int(body.get("correct") or 0))
    except Exception:
        total = correct = 0
    db_.add_quiz_record(user["username"], course_id, total, correct, wrong)
    return {"ok": True}


@app.post("/api/quiz/variant")
def quiz_variant(body: dict, _: dict = Depends(require_user)):
    """答错后「举一反三」：紧扣薄弱知识点再出同类变式题巩固。"""
    course_id = str(body.get("course_id") or "").strip()
    point = str(body.get("point") or "").strip()
    if not course_id or not point:
        raise HTTPException(400, "缺少课程或知识点")
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    from . import quiz as quiz_engine
    n = int(body.get("n") or 3)
    return quiz_engine.generate_variant(graph, point, n)


@app.get("/api/teacher/stats/summary")
def teacher_stats(course_id: str, _: dict = Depends(require_teacher)):
    """教师学情面板：某课程各知识点的错题次数与学生整体掌握情况。"""
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    from . import db as db_
    recs = db_.list_quiz_records(course_id=course_id)
    # 按知识点聚合错题
    point_stat: dict = {}
    students: dict = {}
    for r in recs:
        st = students.setdefault(r["username"], {
            "username": r["username"], "records": 0, "total": 0, "correct": 0,
            "wrong": {}})
        st["records"] += 1; st["total"] += r["total"]; st["correct"] += r["correct"]
        for p in r.get("wrong_points") or []:
            if not p:
                continue
            ps = point_stat.setdefault(p, {
                "name": p, "wrong_count": 0, "students": set()})
            ps["wrong_count"] += 1
            ps["students"].add(r["username"])
            st["wrong"][p] = st["wrong"].get(p, 0) + 1
    # 关联图谱分类
    kind_of = {n.get("name"): n.get("category", "知识点") for n in graph.get("nodes") or []}
    points = [{**ps, "students": sorted(ps["students"]), "category": kind_of.get(ps["name"], "知识点")}
              for ps in point_stat.values()]
    points.sort(key=lambda x: -x["wrong_count"])
    stu_list = []
    for s in students.values():
        s["avg_pct"] = round(s["correct"] * 100 / s["total"]) if s["total"] else 0
        s["wrong"] = [{"name": k, "count": v} for k, v in
                      sorted(s["wrong"].items(), key=lambda kv: -kv[1])]
        stu_list.append(s)
    return {
        "course_id": course_id,
        "summary": {"records": len(recs), "students": len(stu_list)},
        "points": points,
        "students": stu_list,
    }


@app.post("/api/teacher/teaching-plan")
def teaching_plan(body: dict, _: dict = Depends(require_teacher)):
    """教师端：基于当前课程图谱一键生成教学大纲/教案（AIGC 优先，本地兜底）。"""
    course_id = str(body.get("course_id") or "").strip()
    if not course_id:
        raise HTTPException(400, "缺少课程")
    graph = graph_store.get_store().get_graph(course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{course_id}")
    nodes = [n for n in (graph.get("nodes") or []) if n.get("name")]
    cat_order = []
    for n in nodes:
        c = n.get("category", "知识点")
        if c not in cat_order:
            cat_order.append(c)
    name_len = len(nodes)

    def _local_plan():
        from collections import OrderedDict
        groups = OrderedDict()
        for n in nodes:
            groups.setdefault(n.get("category", "其他"), []).append(n)
        lines = [f"# 《{course_id}》教学大纲（由课程知识图谱自动生成）", ""]
        idx = 0
        for cat, ns in groups.items():
            idx += 1
            lines.append(f"## 第{idx}单元 · {cat}")
            for n in ns:
                lines.append(f"  1. {n['name']}"
                             + (f"：{n.get('definition')}" if n.get('definition') else ""))
            lines.append("")
        lines.append("## 教学建议")
        lines.append(f"  · 本课程共识别 {name_len} 个知识点，建议按「{('、'.join(cat_order))}」的顺序组织教学。")
        lines.append("  · 课堂中可结合知识图谱的可视化进行讲解，突出重点概念与前置关系。")
        lines.append("  · 建议配合课后习题，对答错率高的知识点进行专项巩固。")
        return "\n".join(lines)

    mode = "local"
    plan = _local_plan()
    try:
        from . import llm_chat, prompts
        brief = "\n".join(f"- {n.get('name')}（{n.get('category')}）：{n.get('definition') or ''}"
                          for n in nodes[:45])
        from .prompts import truncate
        text, _p = llm_chat.chat(
            [{"role": "system", "content": "你是高校课程教学设计专家，为教师基于课程知识图谱生成一份结构清晰、可直接使用的 Markdown 教学大纲/教案：包含课程简介、教学目标、各单元章节（每章列出核心知识点与学时建议）、考核方式、参考方向。语气专业、条理分明，只输出 Markdown。"},
             {"role": "user", "content":
                 truncate(f"课程名称：{course_id}\n知识点清单：\n{brief or '（暂无）'}", 4000)}],
            provider=["deepseek", "doubao"], temperature=0.4)
        if text and text.strip():
            plan = text.strip()
            mode = "llm"
    except Exception:
        pass
    return {"plan": plan, "course_id": course_id, "mode": mode, "points": name_len}


@app.get("/api/ai/providers")
def ai_providers(_: dict = Depends(require_user)):
    """返回页面内可直接使用的大模型源（豆包 / DeepSeek）及是否已配置真实 Key。"""
    from . import llm_chat
    sup = llm_chat.supported()
    return {
        "providers": [
            {"id": "deepseek", "name": "DeepSeek", "configured": sup.get("deepseek")},
            {"id": "doubao", "name": "豆包", "configured": sup.get("doubao")},
        ],
    }


# ---------- 教师端 AI 工具辅助（仅教师，学生端不展示） ----------

@app.get("/api/assistant/tools")
def assistant_tools(_: dict = Depends(require_teacher)):
    return {"tools": teacher_assistant.tools()}


@app.post("/api/assistant/generate")
def assistant_generate(body: AssistantRequest, _: dict = Depends(require_teacher)):
    graph = graph_store.get_store().get_graph(body.course_id)
    if not graph:
        raise HTTPException(404, f"未找到课程：{body.course_id}")
    return teacher_assistant.generate(body.course_id, graph, body.tool_id, body.task, role="teacher")


# ---------- 教师端：课程文件库（按内容自动分类） ----------

@app.post("/api/teacher/files")
async def upload_files(course_id: str, build: bool = False, files: List[UploadFile] = File(...),
                       user: dict = Depends(require_teacher)):
    added = []
    texts = []
    for f in files:
        raw = await f.read()
        filename = f.filename or "未命名.txt"
        # 任意格式均可上传存档；能解析出文本的才参与自动构图
        parsed = True
        try:
            text = document_parser.extract_text_from_bytes(raw, filename)
        except Exception:
            text = ""
            parsed = False
        klass = classify.classify(text) if parsed else {
            "category": "其他", "label": "已存档（无文本可解析）"}
        fid = db.add_file(user["username"], course_id or "默认课程", filename,
                          klass["category"], klass["label"], len(raw), text)
        added.append({"id": fid, "filename": filename, "category": klass["category"],
                      "label": klass["label"], "size": len(raw), "parsed": parsed})
        if parsed and text.strip():
            texts.append(text)
    result = None
    if build and texts:
        # 自动构图：合并本次上传的所有文件内容，一次性抽取生成课程知识图谱
        result = extractor.extract("\n\n".join(texts))
        graph_store.get_store().save(course_id or "默认课程", result)
        chat_engine.index_course(course_id or "默认课程", result)
    return {
        "added": added,
        "files": db.list_files(user["username"], course_id or "默认课程"),
        "build": result is not None,
        "entities": len(result.entities) if result else 0,
        "relations": len(result.relations) if result else 0,
        "extractor": getattr(extractor, "name", "deepseek") if result else None,
        "graph": graph_store.get_store().get_graph(course_id or "默认课程") if result else None,
    }


@app.get("/api/teacher/files")
def teacher_files(course_id: str = "", user: dict = Depends(require_teacher)):
    return {"files": db.list_files(user["username"], course_id or None)}


@app.delete("/api/teacher/files/{file_id}")
def delete_file(file_id: int, user: dict = Depends(require_teacher)):
    if not db.delete_file(file_id, user["username"]):
        raise HTTPException(404, "文件不存在")
    return {"ok": True}


@app.post("/api/teacher/files/{file_id}/build")
def build_graph_from_file(file_id: int, user: dict = Depends(require_teacher)):
    record = db.get_file(file_id, user["username"])
    if not record:
        raise HTTPException(404, "文件不存在")
    result = extractor.extract(record["text_content"])
    graph_store.get_store().save(record["course_id"], result)
    chat_engine.index_course(record["course_id"], result)
    return {
        "course_id": record["course_id"],
        "extractor": getattr(extractor, "name", "deepseek"),
        "entities": len(result.entities),
        "relations": len(result.relations),
        "graph": graph_store.get_store().get_graph(record["course_id"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
