"""学生端「自动刷题」出题引擎。

- 有可用大模型（DeepSeek/豆包）：要求模型根据课程知识图谱输出结构化单选题（题干/选项/答案/解析）
- 无 Key 或解析失败：基于图谱知识点在本服务内生成 mock 单选题，保证演示不中断
"""
import json
import random

from . import llm_chat
from .prompts import truncate

_SYSTEM = (
    "你是高校课程助教，为学生基于课程知识图谱生成单选题。"
    "严格按照要求输出，每条题目是一个 JSON 对象，字段为："
    '{"q": 题干, "options": [四个选项字符串], "answer": 正确选项的下标(0~3，与options对应), "explain": 解析}。'
    "只能有 4 个选项，且仅一个正确。除 JSON 数组外不要输出任何其他文字、标题或代码块说明。"
)


def _node_brief(graph: dict) -> str:
    nodes = graph.get("nodes") or []
    return "\n".join(
        f"- {n.get('name')}（{n.get('category', '')}）：{n.get('definition') or '暂无定义'}"
        for n in nodes[:40]
    )


def _parse_json_list(text: str):
    """把模型输出解析成题目 JSON 数组，尽量健壮。"""
    if not text:
        return None
    try:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("```", 2)[1]
            clean = clean[1:] if clean.startswith("json") else clean
        data = json.loads(clean)
    except Exception:
        try:
            start = text.find("[")
            end = text.rfind("]")
            data = json.loads(text[start:end + 1]) if start >= 0 and end > start else None
        except Exception:
            return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or item.get("question") or "").strip()
        options = item.get("options") or []
        if not q or not isinstance(options, list) or len(options) < 2:
            continue
        ans = item.get("answer")
        if isinstance(ans, int):  # 下标
            pass
        elif isinstance(ans, str):  # 允许 "A" / "0"
            ans = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ans.upper())
            ans = ans if ans is not None and ans < len(options) else None
        if ans is None or ans < 0 or ans >= len(options):
            continue
        out.append({
            "q": q,
            "options": [str(o).strip() for o in options],
            "answer": int(ans),
            "explain": (item.get("explain") or "").strip() or "（该题未提供解析）",
        })
    return out or None


def _llm_quiz(graph: dict, n: int):
    brief = _node_brief(graph)
    user = (
        f"课程知识点：\n{brief or '（课程暂无知识点）'}\n\n"
        f"请基于以上知识点生成 {n} 道单选题，考察对核心概念的掌握。"
        "答案与解析都要来自课程知识点，不要编造知识点以外的概念。"
    )
    try:
        text, _provider = llm_chat.chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": truncate(user, 4000)},
            ],
            provider=["deepseek", "doubao"],
            temperature=0.5,
        )
    except Exception:
        return None
    if not text:
        return None
    return _parse_json_list(text)


def _mock_quiz(graph: dict, n: int):
    """基于图谱知识点的 mock 单选题，保证离线/无 Key 也可刷题。"""
    nodes = [x for x in (graph.get("nodes") or []) if x.get("name")]
    if not nodes:
        return []
    random.shuffle(nodes)
    chosen = nodes[:n]
    defs = [x.get("definition") for x in nodes if x.get("definition")]
    generic = [
        "与课程无关的说法",
        "在课程中并未被定义",
        "以上说法都不准确",
        "它是本课程的前置又与进阶内容相关",
    ]
    items = []
    for nd in chosen:
        name = nd["name"]
        cat = nd.get("category", "知识点")
        correct = nd.get("definition") or f"「{name}」是课程中的一个「{cat}」类知识点。"
        opts = [correct]
        pool = list(defs)
        random.shuffle(pool)
        for d in pool:
            if d and d != correct and d not in opts:
                opts.append(d)
            if len(opts) == 4:
                break
        gi = 0
        while len(opts) < 4 and gi < len(generic):
            if generic[gi] not in opts:
                opts.append(generic[gi])
            gi += 1
        random.shuffle(opts)
        ans = opts.index(correct)
        items.append({
            "q": f"关于「{name}」（{cat}）的定义，下列说法最准确的是？",
            "options": opts,
            "answer": ans,
            "explain": f"正确理解：{correct} 它属于课程中的「{cat}」类知识点。",
        })
    return items


def generate_quiz(graph: dict, n: int = 5):
    n = max(1, min(int(n or 5), 10))
    mode = "mock"
    qs = None
    try:
        if llm_chat.supported().get("deepseek") or llm_chat.supported().get("doubao"):
            qs = _llm_quiz(graph, n)
            mode = "llm" if qs else "mock"
    except Exception:
        qs = None
    if not qs:
        qs = _mock_quiz(graph, n)
    return {"questions": qs, "mode": mode}