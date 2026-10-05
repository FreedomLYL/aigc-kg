"""课后习题、作业布置、学情分析与教案生成引擎（教师端 & 学生端）。

- 出题：优先调用真实大模型，按其布置要求 / 巩固知识点生成选择题；无 Key 时按图谱知识点模板出题
- 作业布置：教师针对课程保存布置要求，学生端据此出题
- 变式巩固：针对某一薄弱知识点再出若干变式题
- 学情分析：基于学生作答记录统计各知识点薄弱程度
- 教案生成：基于课程图谱生成结构化 Markdown 教案（大模型优先，本地模板兜底）
"""
import json
import random
import re

from . import llm_chat


class QuizEngine:
    """基于课程知识图谱的智能出题 / 变式 / 教案生成引擎。"""

    def __init__(self):
        self._use_llm = bool(llm_chat.supported().get("doubao") or llm_chat.supported().get("deepseek"))

    # ---------- 出题 ----------

    def generate(self, graph: dict, n: int, requirement: str = "") -> dict:
        """为课程生成 n 道题。requirement 为教师布置要求（可为空）。返回 questions + mode。"""
        nodes = graph.get("nodes", [])
        if not nodes:
            return {"questions": [], "mode": "template"}
        n = max(1, min(int(n or 5), 50))
        if requirement and requirement.strip():
            questions, mode = self._generate_llm(
                nodes, n, "作业要求：" + requirement.strip(), temperature=0.5)
            if questions:
                return {"questions": questions, "mode": mode}
        return {"questions": self._template_questions(nodes, n), "mode": "template"}

    def variant(self, graph: dict, point: str, n: int) -> dict:
        """针对某个薄弱知识点出 n 道变式巩固题。"""
        nodes = graph.get("nodes", [])
        n = max(1, min(int(n or 3), 10))
        if not nodes:
            return {"questions": [], "mode": "template"}
        questions, mode = self._generate_llm(
            nodes, n, f"围绕知识点「{point}」出 {n} 道侧重概念辨析的变式巩固题", temperature=0.5)
        if questions:
            return {"questions": questions, "mode": mode}
        return {"questions": self._template_questions(nodes, n, focus=point), "mode": "template"}

    # ---------- 教案生成 ----------

    def teaching_plan(self, graph: dict) -> dict:
        """基于课程图谱生成结构化 Markdown 教案；大模型优先，本地模板兜底。"""
        nodes = graph.get("nodes", [])
        if not nodes:
            return {"plan": "", "points": 0, "mode": "template"}
        plan, mode = self._plan_llm(nodes)
        if plan:
            return {"plan": plan, "points": len(nodes), "mode": mode}
        return {"plan": self._plan_template(graph, nodes), "points": len(nodes), "mode": "template"}

    # ---------- 学情统计 ----------

    @staticmethod
    def stats(records: list) -> dict:
        """基于学生作答记录统计课程学情。传入 db.list_quiz_records() 结果。"""
        if not records:
            return {"summary": {"records": 0, "students": 0}, "points": [], "students": []}
        point_count = {}
        for r in records:
            for pt in r.get("wrong_points", []):
                pt = str(pt).strip()
                if pt:
                    point_count[pt] = point_count.get(pt, 0) + 1
        points = [{"name": k, "wrong_count": v} for k, v in
                  sorted(point_count.items(), key=lambda x: -x[1])]
        per_user = {}
        for r in records:
            u = r["username"]
            if u not in per_user:
                per_user[u] = {"records": 0, "total": 0, "correct": 0}
            per_user[u]["records"] += 1
            per_user[u]["total"] += int(r.get("total") or 0)
            per_user[u]["correct"] += int(r.get("correct") or 0)
        students = []
        for u, s in per_user.items():
            total = s["total"]
            avg = round(s["correct"] * 100 / total) if total else 0
            students.append({"username": u, "records": s["records"], "avg_pct": avg})
        students.sort(key=lambda x: (-x["records"], -x["avg_pct"]))
        return {
            "summary": {"records": len(records), "students": len(students)},
            "points": points,
            "students": students,
        }

    # ---------- 内部实现 ----------

    @staticmethod
    def _node_brief(nodes) -> str:
        return "\n".join(
            f"- {n.get('name', '')}（{n.get('category', '概念')}）：{n.get('definition') or '暂无定义'}"
            for n in nodes[:50])

    def _generate_llm(self, nodes, n, task, temperature=0.5):
        """调用大模型生成带解析的选择题 JSON；失败返回 (None, None)。"""
        if not self._use_llm:
            return None, None
        brief = self._node_brief(nodes)
        system = (
            "你是课程测验出题助手。依据课程知识图谱知识点，生成评测学生掌握度的单项选择题。\n"
            "要求：1) 每题 4 个选项；2) answer 必须是正确选项的 0 基索引（0~3）；"
            "3) 每题 point 字段标注对应知识点名称；4) explain 给出解析；"
            "5) 只输出 JSON，不要解释文字。"
        )
        user = f"课程知识点：\n{brief}\n\n出题任务：{task}"
        text, _ = llm_chat.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"请生成 {n} 道题（不少于 4 道，若知识点不足可减少）。\n"
                                            f"任务：{task}\n知识点：\n{brief[:3000]}"},
            ],
            provider="deepseek", temperature=temperature,
        )
        if not text:
            return None, None
        try:
            data = json.loads(re.sub(r"```json|```", "", text).strip())
        except Exception:
            return None, None
        questions = []
        items = data if isinstance(data, list) else (data.get("questions") or [])
        for it in items:
            if not isinstance(it, dict):
                continue
            options = [str(o) for o in (it.get("options") or [])]
            if len(options) < 2:
                continue
            q = {
                "q": str(it.get("q", "")),
                "options": options,
                "answer": int(it.get("answer", 0)) % len(options),
                "explain": str(it.get("explain", "")),
                "point": str(it.get("point", "")),
            }
            if q["q"]:
                questions.append(q)
            if len(questions) >= n:
                break
        return (questions, "llm") if questions else (None, None)

    @staticmethod
    def _template_questions(nodes, n, focus=None):
        """兜底：按图谱知识点模板生成单选择题。focus 指定需确保的薄弱知识点。"""
        pool = [nd for nd in nodes if nd.get("name")]
        if not pool:
            return []
        # 组装 (question_text, options, answer_idx, correct_str, point, explain)
        built = []
        used_names = set()
        if focus:
            hit = next((nd for nd in pool if nd["name"] == focus), None)
            if hit:
                built.append(QuizEngine._make_from_node(hit, pool))
                used_names.add(hit["name"])
        for nd in pool:
            if nd["name"] in used_names:
                continue
            built.append(QuizEngine._make_from_node(nd, pool))
            used_names.add(nd["name"])
            if len(built) >= n:
                break
        if not built:
            return []
        random.shuffle(built)
        return built[:n]

    @staticmethod
    def _make_from_node(nd, pool):
        name = nd.get("name", "")
        definition = nd.get("definition") or "是这门课程中的一个重要知识点"
        kind = nd.get("category", "概念")
        distractors = []
        for o in pool:
            if o.get("name") == name:
                continue
            d = f"{o.get('name', '某知识点')}的关键特征是「{o.get('definition') or '课时内容'}」。"
            distractors.append(d)
            if len(distractors) >= 3:
                break
        options = [definition] + distractors
        random.shuffle(options)
        answer = options.index(definition)
        return {
            "q": f"下列关于「{name}」（{kind}）的说法，最准确的是？",
            "options": options,
            "answer": answer,
            "explain": f"「{name}」：{definition}",
            "point": name,
        }

    def _plan_llm(self, nodes):
        if not self._use_llm:
            return None, None
        brief = self._node_brief(nodes)
        system = (
            "你是高校课程教学设计专家。依据课程知识图谱知识点，编写一份结构化的 Markdown 教学大纲/教案。"
            "应包含：教学目标、教学重难点、课时安排、每节讲授要点、随堂练习与课后作业建议。"
            "只输出 Markdown，不要额外解释。"
        )
        text, _ = llm_chat.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"课程主要知识点：\n{brief[:3000]}"},
            ],
            provider="deepseek", temperature=0.5,
        )
        return (text, "llm") if text else (None, None)

    @staticmethod
    def _plan_template(graph, nodes):
        names = [nd.get("name", "") for nd in nodes if nd.get("name")]
        links = graph.get("links", [])
        pre = [l for l in links if l.get("type") == "pre"]
        steps = list(dict.fromkeys([l["source"] for l in pre] + names[:8]))
        order = " → ".join(n for n in steps[:10])
        lines = [
            f"# {graph.get('course_id') or '本课程'} 教学大纲",
            "",
            "## 一、教学目标",
            f"1. 掌握本课程核心知识点（共 {len(names)} 个），理解知识间的逻辑关系。",
            f"2. 重点理解：{'、'.join(names[:5])}。",
            "3. 能够运用所学知识解决实际问题，并通过随堂练习巩固。",
            "",
            "## 二、教学重难点",
            f"- 重点：{'、'.join(names[:5])}",
            f"- 难点：{'、'.join(names[:3]) + '（建议结合前置关系分步讲解）'}",
            "",
            "## 三、推荐讲授路线（按前置关系）",
            order if order else "见知识图谱导览",
            "",
            "## 四、课时安排建议",
        ]
        for i, nm in enumerate(names[:8], 1):
            lines.append(f"{i}. 第 {i} 节：{nm}")
        lines += [
            "",
            "## 五、随堂练习与作业",
            "在 AI 助教/课后习题中按知识点出题，学生逐题作答即时判分；错题自动记为薄弱点，可举一反三巩固。",
            "",
            "> 本教案由系统根据课程知识图谱自动生成（本地模板；配置大模型 Key 后为真实 AI 生成）。",
        ]
        return "\n".join(lines)


def build_quiz_engine() -> QuizEngine:
    return QuizEngine()