"""教师端 AI 工具辅助引擎。

- 内置常用 AI 工具清单（豆包/飞书/WorkBuddy/Trae/ima/DeepSeek），教师下拉选择
- 在系统网页内直接向所选品牌模型提问（豆包→豆包、DeepSeek→DeepSeek，页面内生成，不跳转外部）
- 有对应 API Key → 调用大模型根据课程图谱 + 教学任务生成内容
- 无 API Key      → Mock 模板（按任务类型：测验题/教案/重难点/通用建议）
"""
from . import llm_chat
from .prompts import truncate


AI_TOOLS = [
    {"id": "doubao", "name": "豆包", "icon": "豆", "color": "#2f6bff",
     "engine": "doubao", "real": True,
     "desc": "字节跳动大模型：多模态问答、写作、办公辅助，页面内直接调用豆包模型（火山方舟）。"},
    {"id": "deepseek", "name": "DeepSeek", "icon": "D", "color": "#4f6bff",
     "engine": "deepseek", "real": True,
     "desc": "开源大模型：通用对话、推理与内容生成，页面内直接调用 DeepSeek 开放平台模型。"},
]

# 路由到真实品牌模型；未命中时回退到另一可用模型
_TOOL_PROVIDER = {"doubao": "doubao", "deepseek": "deepseek"}


class TeacherAssistant:
    def __init__(self):
        self._use_llm = bool(llm_chat.supported().get("doubao") or llm_chat.supported().get("deepseek"))

    def tools(self):
        return AI_TOOLS

    def generate(self, course_id: str, graph: dict, tool_id: str, task: str, role: str = "teacher") -> dict:
        tool = next((t for t in AI_TOOLS if t["id"] == tool_id), AI_TOOLS[0])
        is_student = role == "student"
        nodes = graph.get("nodes", [])
        node_brief = "\n".join(
            f"- {n['name']}（{n.get('category', '')}）：{n.get('definition') or '暂无定义'}"
            for n in nodes[:40]
        )
        fallback_task = "为这份课程梳理一份学习要点与练习题" if is_student else "请为该课程梳理一份课堂教学建议"
        blank_task = (task or "").strip() or fallback_task

        text, provider, mode = None, None, "mock"
        if self._use_llm:
            text, provider = self._llm_generate(tool, blank_task, node_brief, is_student)
            mode = "llm" if text else "mock"
        if not text:
            text = self._mock_generate(tool, blank_task, nodes, is_student)

        return {
            "tool": {"id": tool["id"], "name": tool["name"], "desc": tool["desc"]},
            "scope": "learning" if is_student else "teaching",
            "mode": mode,
            "provider": provider,  # 实际使用的模型品牌（doubao/deepseek），None 表示演示模板
            "generated": text,
        }

    def _llm_generate(self, tool: dict, task: str, node_brief: str, is_student: bool):
        """按工具路由到品牌模型；未命中品牌的工具回退顺序 = DeepSeek→豆包。"""
        if is_student:
            system_prompt = (
                f"你正在以「{tool['name']}」的身份，充当一位课程 AI 学习教练，辅导学生自主学习本课程。"
                "请依据下面从课程知识图谱中抽取的知识点，围绕学生给出的学习需求，产出通俗、可读、有步骤的内容"
                "（如知识点讲解/解题思路/随堂小练习/学习小结均可）。不要编造知识点以外的概念。"
            )
            user_prompt = f"课程知识点：\n{node_brief or '（课程暂无知识点）'}\n\n学生学习需求：{task}"
        else:
            system_prompt = (
                f"你正在以「{tool['name']}」的身份，辅助一位高校教师备课。"
                "请依据下面从课程知识图谱中抽取的知识点，围绕教师给出的教学任务产出直接可用、结构清晰的文本"
                "（如测验题/教案/重难点/教学设计均可）。不要编造知识点以外的概念。"
            )
            user_prompt = f"课程知识点：\n{node_brief or '（课程暂无知识点）'}\n\n教师任务：{task}"
        order = [_TOOL_PROVIDER.get(tool["id"], "deepseek")]
        for p in ("deepseek", "doubao"):
            if p not in order:
                order.append(p)
        return llm_chat.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": truncate(user_prompt, 4000)},
            ],
            provider=order,
            temperature=0.6,
        )

    @staticmethod
    def _mock_generate(tool: dict, task: str, nodes, is_student: bool = False) -> str:
        names = [n["name"] for n in (nodes or [])]
        t = task
        lines = [f"✅ 已选用 AI 工具：**{tool['name']}**（{tool['desc']}）"]
        lines.append(f"\n【{'学习需求' if is_student else '教学任务'}】{t}")

        if any(k in t for k in ("题", "测验", "试题", "选择", "试卷")):
            lines.append("\n【建议·随堂测验】")
            for i, n in enumerate(names[:5], 1):
                lines.append(f"{i}. 单选题：关于「{n}」，下列说法最准确的是（ ）\n   A. 符合课程定义  B. 与课程无关  C. 二者皆可  D. 无法判断\n   答案：A（以课程图谱定义为准）")
            lines.append("\n可让 AI 工具结合每题对应知识点的定义自动生成干扰项与解析。")
        elif any(k in t for k in ("教案", "教学设计", "大纲", "备课")):
            lines.append("\n【建议·教学设计】\n一、教学目标\n二、教学重难点\n三、教学过程：导入 → 知识讲解 → 随堂练习 → 小结\n四、课后作业")
            lines.append("\n主线知识点（按图谱）：" + " → ".join(names[:8]))
            lines.append("\n可让 AI 工具依据以上知识点自动扩写每节的讲授要点与例题。")
        elif any(k in t for k in ("重难点", "重点", "难点")):
            lines.append("\n【建议·重难点】")
            lines.append("1. 核心概念（图谱中的概念/术语节点）：" + "、".join(names[:5]))
            lines.append("2. 建议结合前置关系讲解，先夯实基础概念再讲算法与公式。")
        else:
            lines.append("\n【建议·通用教学建议】")
            lines.append("1. 结合知识图谱分块讲授，先概念（" + "、".join(names[:3]) + "）再方法/算法。")
            lines.append("2. 用图谱问答让学生自查理解，布置小组互讲巩固。")
            lines.append("3. 定期重新抽取/校对图谱，保持知识点与课程同步。")
        lines.append("\n—— 由「" + tool["name"] + "」辅助生成（当前为演示模板；配置 DeepSeek/豆包 Key 后为真实 AI 生成）")
        return "\n".join(lines)


def build_assistant() -> TeacherAssistant:
    return TeacherAssistant()