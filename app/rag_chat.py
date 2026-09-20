"""RAG 智能问答引擎。

- 检索课程知识图谱知识点
- 有可用大模型 API Key → 在系统网页内直接向所选品牌模型（豆包/DeepSeek）生成自然语言回答
- 无 API Key            → 检索知识点 + 模板拼接，仍优于纯关键词匹配
"""
import json
import re
from typing import Dict, List, Optional

from . import llm_chat
from .schemas import ExtractionResult

try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


class SimpleRetriever:
    """轻量检索器：基于 TF-IDF 风格的关键词匹配 + 编辑距离兜底。

    无需外部向量库，零依赖即可运行。接入真实向量库时可替换此类。
    """

    def __init__(self):
        self._index: Dict[str, List[dict]] = {}

    def index(self, course_id: str, extraction: ExtractionResult) -> None:
        docs = []
        for ent in extraction.entities:
            text = f"{ent.name} {ent.kind} {ent.definition or ''}"
            docs.append({
                "name": ent.name,
                "kind": ent.kind,
                "definition": ent.definition or "",
                "text": text,
            })
        for rel in extraction.relations:
            for doc in docs:
                if doc["name"] in (rel.source, rel.target):
                    other = rel.target if rel.source == doc["name"] else rel.source
                    doc["text"] += f" {other} {rel.type}"
        self._index[course_id] = docs

    def index_from_graph(self, course_id: str, graph: dict) -> None:
        """从图谱 JSON 重建索引（编辑后调用）。"""
        docs = []
        for node in graph.get("nodes", []):
            docs.append({
                "name": node["name"],
                "kind": node.get("category", ""),
                "definition": node.get("definition", ""),
                "text": f"{node['name']} {node.get('category', '')} {node.get('definition', '')}",
            })
        for link in graph.get("links", []):
            for doc in docs:
                if doc["name"] in (link["source"], link["target"]):
                    other = link["target"] if link["source"] == doc["name"] else link["source"]
                    doc["text"] += f" {other} {link.get('type', '')}"
        self._index[course_id] = docs

    def retrieve(self, course_id: str, query: str, top_k: int = 5) -> List[dict]:
        docs = self._index.get(course_id, [])
        if not docs:
            return []
        query_terms = self._tokenize(query)
        scored = []
        for doc in docs:
            score = 0
            doc_terms = self._tokenize(doc["text"])
            for qt in query_terms:
                for dt in doc_terms:
                    if qt == dt:
                        score += 2
                    elif qt in dt or dt in qt:
                        score += 1
                if qt in doc["name"]:
                    score += 3
            scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored[:top_k] if score > 0]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
        return tokens


class RAGChatEngine:
    """RAG 问答引擎：检索 + 生成（支持在页面内选择模型源）。"""

    def __init__(self):
        self._retriever = SimpleRetriever()
        from . import config
        self._use_llm = bool(llm_chat.supported().get("doubao") or llm_chat.supported().get("deepseek"))

    def index_course(self, course_id: str, extraction: ExtractionResult) -> None:
        self._retriever.index(course_id, extraction)

    def _ensure_index(self, course_id: str, graph: dict) -> None:
        if course_id not in self._retriever._index:
            self._retriever.index_from_graph(course_id, graph)

    def ask(self, course_id: str, question: str, graph: dict, history: Optional[List[dict]] = None,
            provider="deepseek") -> dict:
        self._ensure_index(course_id, graph)
        retrieved = self._retriever.retrieve(course_id, question, top_k=5)
        history = history or []

        if not retrieved:
            fallback = self._graph_fallback(question, graph)
            if fallback:
                name_hit = any(n["name"] in question for n in fallback)
                if name_hit:
                    lines = [f"根据当前课程图谱，与「{question}」直接相关的知识点："]
                    lines += [f"· {n['name']}（{n.get('category', '概念')}）：{n.get('definition', '见图谱')}" for n in fallback]
                else:
                    lines = ["这是当前课程的重点速览（来自知识图谱），看看你从哪开始："]
                    lines += [f"· {n['name']}（{n.get('category', '概念')}）：{n.get('definition', '见图谱')}" for n in fallback]
                lines.append("\n可以继续问：某个知识点的含义、知识点之间的关系，或让我出一道练习题。")
                return {
                    "answer": "\n".join(lines),
                    "references": [n["name"] for n in fallback],
                    "mode": "mock",
                    "provider": None,
                    "suggested": self._suggest_questions(fallback, graph),
                }
            return {
                "answer": "图谱中未找到与问题直接相关的知识点，请换个问法或先构建图谱。",
                "references": [],
                "mode": "none",
                "provider": None,
                "suggested": self._suggest_questions(retrieved=None, graph=graph),
            }

        references = [doc["name"] for doc in retrieved]
        is_followup = bool(history)

        if self._use_llm:
            answer, used = self._generate(question, retrieved, history, provider)
            mode = "llm" if answer else "mock"
        else:
            answer = self._template_answer(question, retrieved, history, is_followup)
            used = None
            mode = "mock"

        return {
            "answer": answer,
            "references": references,
            "mode": mode,
            "provider": used,
            "suggested": self._suggest_questions(retrieved, graph),
        }

    def _graph_fallback(self, question: str, graph: dict) -> Optional[List[dict]]:
        """检索为空时，用图谱节点做兜底：问题包含节点名优先；否则"总结/重点/大纲"类提问给核心节点速览。"""
        nodes = graph.get("nodes", [])
        if not nodes:
            return None
        # 1) 问题文本里出现了某个节点名 → 精确命中
        named = [n for n in nodes if n.get("name") and n["name"] in question]
        if named:
            return named
        # 2) 总结/概览类提问 → 用入出度最高的核心节点做速览
        if any(k in question for k in ("总结", "重点", "大纲", "本章", "介绍", "概览", "学什么")):
            import collections
            deg = collections.Counter()
            for link in graph.get("links", []):
                deg[link["source"]] += 1
                deg[link["target"]] += 1
            ranked = sorted(nodes, key=lambda n: -deg.get(n["name"], 0))
            return ranked[:6]
        return None

    def _suggest_questions(self, retrieved: Optional[List[dict]], graph: dict) -> List[str]:
        """根据图谱/检索结果，给出 3 条可点击的后续提问。"""
        if retrieved:
            names = [d["name"] for d in retrieved[:3]]
            return [f"再详细讲解一下「{names[0]}」",
                    f"「{names[0]}」和「{names[1] if len(names) > 1 else '它'}」是什么关系？",
                    f"围绕「{names[0]}」出一道练习题"]
        nodes = graph.get("nodes", [])
        picks = [n["name"] for n in nodes[:3]]
        if picks:
            return [f"「{picks[0]}」是什么意思？" for p in picks]
        return ["帮我总结本章重点", "我该如何开始学习？"]

    def _generate(self, question: str, retrieved: List[dict], history: Optional[List[dict]],
                  provider: str):
        context_parts = []
        for doc in retrieved:
            context_parts.append(
                f"- {doc['name']}（{doc['kind']}）：{doc['definition']}"
            )
        context = "\n".join(context_parts)

        system_prompt = (
            "你是一名课程知识助教。根据下方从课程知识图谱中检索到的知识点，并结合对话历史，回答学生的问题。"
            "要求：1) 只基于提供的知识点回答，不要编造；"
            "2) 若学生是追问（对话中已有上下文），结合上文连贯作答；"
            "3) 语言简洁清晰；4) 知识点不足以回答时，说明并建议学生查看相关内容。"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:  # 最多带最近 3 轮上下文（兼容 dict 与对象）
            if isinstance(turn, dict):
                role, content = turn.get("role", "user"), turn.get("content", "")
            else:
                role, content = getattr(turn, "role", "user"), getattr(turn, "content", "")
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": content})
        messages.append({
            "role": "user",
            "content": f"检索到的知识点：\n{context}\n\n学生问题：{question}",
        })

        order = [p for p in (provider, "deepseek", "doubao") if p in ("doubao", "deepseek")]
        text, used = llm_chat.chat(messages, provider=order, temperature=0.3)
        if text:
            return text, used
        return self._template_answer(question, retrieved, history, bool(history)), None

    @staticmethod
    def _template_answer(question: str, retrieved: List[dict], history: Optional[List[dict]], is_followup: bool) -> str:
        context = "\n".join(
            f"· {doc['name']}（{doc['kind']}）：{doc['definition']}" for doc in retrieved
        )
        if is_followup:
            lines = [f"好的，结合我们刚刚聊的内容，关于「{question}」再补充说明：", context]
            lines.append("\n如果想深入，可以继续追问，或点击图谱中的知识点查看详情。")
        elif any(k in question for k in ("题", "练习", "出")):
            lines = ["这是一道基于课程知识点的练习：", ""]
            for i, doc in enumerate(retrieved[:3], 1):
                lines.append(f"{i}. 根据「{doc['name']}」的定义（{doc['definition'] or '见图谱'}），请用自己的话说说它解决什么问题，并举一个例子。")
            lines.append("\n（参考答案要点见上方的知识点定义。）")
        else:
            lines = ["根据课程知识图谱，以下是与你的问题最相关的知识点：", context]
            lines.append("\n还可以继续问：解释某一知识点、它与前序知识点的联系、或者让我出一道练习题。")
        return "\n".join(lines)


def build_chat_engine() -> RAGChatEngine:
    return RAGChatEngine()
