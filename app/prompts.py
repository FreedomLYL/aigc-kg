"""提示词工程：知识抽取所需的核心 system 提示词。

要点：
- 强制 JSON 输出，便于程序稳定解析
- 明确实体/关系结构
- 关系类型使用受控枚举（前置/包含/相关/应用/对比）
"""

KNOWLEDGE_EXTRACTION_SYSTEM = """你是一名课程知识图谱构建助手。给定一段课程章节文本，请抽取出其中的知识点实体及其之间的语义关系。

要求：
1. 实体(kind 取其一)：概念、公式、定理、算法、术语、方法
2. relationship 仅使用：前置(pre)、包含(has_part)、相关(related)、应用(applies_to)、对比(contrast)
   - 前置：学习 B 之前应当先掌握 A，则 A->B 记 pre
   - 包含：A 是 B 的一部分
   - 相关：两者概念相关但非前述关系
3. 实体抽取宁缺毋滥，聚焦真正重要的知识点
4. 必须只输出 JSON，不要任何解释文字

输出 JSON 格式如下：
{
  "entities": [
    {"name": "知识点名", "kind": "概念", "definition": "一句话定义"}
  ],
  "relations": [
    {"source": "源知识点", "target": "目标知识点", "type": "pre|has_part|related|applies_to|contrast"}
  ]
}
"""


def truncate(text: str, max_chars: int = 6000) -> str:
    """避免超出上下文长度，按字符截断。"""
    return text if len(text) <= max_chars else text[:max_chars]