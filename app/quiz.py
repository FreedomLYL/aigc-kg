"""学生端「自动刷题」出题引擎。

- 有可用大模型（DeepSeek/豆包）：要求模型根据课程知识图谱输出结构化单选题（题干/选项/答案/解析）
- 无 Key 或解析失败：基于图谱知识点在本服务内生成 mock 单选题，保证演示不中断
"""
import json
import random
import re

from . import llm_chat
from .prompts import truncate

_SYSTEM = (
    "你是高校课程助教，为学生基于课程知识图谱生成单选题。"
    "严格按照要求输出，每条题目是一个 JSON 对象，字段为："
    '{"q": 题干, "options": [四个选项字符串], "answer": 正确选项的下标(0~3，与options对应), "explain": 解析, "point": 本题考察的知识点名称(必须来自给出的课程知识点列表，精确匹配一个名称)}。'
    "只能有 4 个选项，且仅一个正确。除 JSON 数组外不要输出任何其他文字、标题或代码块说明。"
)


def _node_brief(graph: dict) -> str:
    nodes = graph.get("nodes") or []
    return "\n".join(
        f"- {n.get('name')}（{n.get('category', '')}）：{n.get('definition') or '暂无定义'}"
        for n in nodes[:40]
    )


def _align_points(graph: dict, questions: list) -> list:
    """把每题 point 校正为图谱中的真实节点名。

    大模型可能把 point 写成变体（如“深度学习（概念）”），与图谱节点名“深度学习”不一致，
    会影响薄弱点归并与举一反三。这里把 point 对齐到真实节点名，无法对齐则留空。
    """
    names = [n.get("name") for n in (graph.get("nodes") or []) if n.get("name")]
    if not names:
        return questions
    strip = lambda s: re.sub(r"[（(【].*?[)】)]", "", s or "").strip()
    by_strip = {}
    for n in names:
        by_strip.setdefault(strip(n), n)
    for it in questions:
        p = str(it.get("point") or "").strip()
        if not p:
            continue
        if p in names:
            continue
        hit = by_strip.get(p) or by_strip.get(strip(p))
        if hit:
            it["point"] = hit
            continue
        q = str(it.get("q") or "")
        cand = next((n for n in names if n and n in q), None)
        it["point"] = cand or ""
    return questions


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
            "point": (item.get("point") or "").strip(),
        })
    return out or None


def _llm_quiz(graph: dict, n: int, requirement: str = ""):
    brief = _node_brief(graph)
    req_part = ""
    if requirement and requirement.strip():
        req_part = f"\n\n教师的布置要求（必须严格执行）：{requirement.strip()}"
    user = (
        f"课程知识点：\n{brief or '（课程暂无知识点）'}\n\n"
        f"请基于以上知识点生成 {n} 道单选题，考察对核心概念的掌握。"
        "答案与解析都要来自课程知识点，不要编造知识点以外的概念。"
        f"{req_part}"
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
            "point": name,
        })
    return items


def generate_quiz(graph: dict, n: int = 5, requirement: str = ""):
    n = max(1, min(int(n or 5), 50))
    mode = "mock"
    qs = None
    try:
        if llm_chat.supported().get("deepseek") or llm_chat.supported().get("doubao"):
            qs = _llm_quiz(graph, n, requirement)
            mode = "llm" if qs else "mock"
            if qs:
                qs = _align_points(graph, qs)
    except Exception:
        qs = None
    if not qs:
        qs = _mock_quiz(graph, n)
    if qs and len(qs) > n:
        qs = qs[:n]
    return {"questions": qs, "mode": mode}


def generate_variant(graph: dict, point: str, n: int = 3):
    """答错后「举一反三」：紧扣某个薄弱知识点再出同类变式题。"""
    n = max(1, min(int(n or 3), 8))
    if not point:
        return generate_quiz(graph, n)
    # 确认知识点真实存在于图谱，避免无效出题
    nodes = graph.get("nodes") or []
    real = next((x for x in nodes if x.get("name") == point), None)
    if not real:
        return generate_quiz(graph, n)
    mode = "mock"
    qs = None
    try:
        if llm_chat.supported().get("deepseek") or llm_chat.supported().get("doubao"):
            brief = _node_brief(graph)
            user = (
                f"课程知识点（重点：{point}）：\n{brief or '（课程暂无知识点）'}\n\n"
                f"请围绕知识点「{point}」的变化应用、易错辨析、概念延伸出 {n} 道"
                "变式单选题，尽量从不同角度考察同一个点，帮助复习巩固。"
                "每题 point 字段必须填「" + point + "」。"
            )
            try:
                text, _p = llm_chat.chat(
                    [{"role": "system", "content": _SYSTEM},
                     {"role": "user", "content": truncate(user, 4000)}],
                    provider=["deepseek", "doubao"], temperature=0.6,
                )
            except Exception:
                text = None
            if text:
                qs = _parse_json_list(text)
                if qs:
                    for it in qs:
                        it["point"] = point
                    mode = "llm"
    except Exception:
        qs = None
    if not qs:
        # 本地变式：围绕该知识点的定义/辨析措辞换新
        items = []
        base = [real.get("definition") or f"「{point}」是课程知识点。"]
        for i in range(n):
            ring = "判断正误" if i % 2 == 0 else "选择最准确的说法"
            correct = base[0]
            others = [x.get("definition") for x in nodes
                      if x.get("definition") and x.get("definition") != correct][:3]
            opts = [correct] + others
            while len(opts) < 4:
                opts.append(f"与「{point}」无关或自相矛盾的说法（变式{i}）")
            opts = opts[:4]
            random.shuffle(opts)
            items.append({
                "q": f"（举一反三 · 第{i+1}题）关于「{point}」，{ring}？",
                "options": opts,
                "answer": opts.index(correct),
                "explain": f"复习要点：{correct} 这仍是围绕「{point}」的知识点。",
                "point": point,
            })
        qs = items
    if len(qs) > n:
        qs = qs[:n]
    return {"questions": qs, "mode": mode}