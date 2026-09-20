"""课程文件按内容自动分类（教师端）。

基于关键词命中计分的确定性规则分类器，无需 API Key，适用于常见中文课程文档。
从上传文档抽取出的文本中统计各类别关键词命中数，取分值最高项；平票按类别优先级裁决。
"""
import re
from typing import Dict, List, Tuple

# 类别定义：key -> (中文标签, 关键词列表, 优先级(数字越大越优先裁决平票))
CATEGORY_RULES: Dict[str, Tuple[str, List[str], int]] = {
    "syllabus": ("教学大纲", [
        "教学大纲", "课程大纲", "教学计划", "考核方式", "课程目标", "学时分配",
        "教学安排", "课程序言", "syllabus", "outline", "objectives",
    ], 6),
    "slides": ("课件", [
        "第.*章", "第一节", "本讲", "课件", "本章要点", "复习要点", "内容提要",
        "slide", "lecture", "presentation",
    ], 5),
    "notes": ("讲义", [
        "定义", "定理", "公式", "推导", "证明", "例如", "即", "设", "那么",
        "note", "lecture notes", "handout",
    ], 4),
    "assignments": ("习题/作业", [
        "习题", "练习", "作业", "思考题", "计算题", "解答", "答案", "布置",
        "exercise", "assignment", "homework", "practice",
    ], 3),
    "exams": ("试卷/测验", [
        "期末", "期中", "试卷", "测验", "选择", "填空", "简答", "论述", "得分",
        "满分", "考试", "exam", "test", "quiz", "answer sheet",
    ], 3),
    "lab": ("实验", [
        "实验", "仪器", "实验步骤", "实验数据", "操作", "记录", "结果分析",
        "lab", "experiment",
    ], 2),
    "reference": ("参考/拓展", [
        "参考文献", "参考资料", "拓展阅读", "延伸阅读", "附录", "推荐书目",
        "reference", "appendix", "further reading",
    ], 2),
}

CATEGORY_LABEL = {k: v[0] for k, v in CATEGORY_RULES.items()}


def _count(text: str, patterns: List[str]) -> int:
    score = 0
    for p in patterns:
        try:
            score += len(re.findall(p, text, re.IGNORECASE))
        except re.error:
            score += text.lower().count(p.lower())
    return score


def classify(text: str) -> Dict[str, object]:
    """返回 {category, label, score}。文本过短/无任何命中时归为 其他。"""
    sample = text[:4000]
    scored: List[Tuple[str, int, int]] = []  # (key, score, priority)
    for key, (_label, patterns, priority) in CATEGORY_RULES.items():
        c = _count(sample, patterns)
        if c > 0:
            scored.append((key, c, priority))

    if not scored:
        return {"category": "other", "label": "其他", "score": 0}

    # 先按分数降序，再按优先级（分数相同时取更具体类别）
    best = max(scored, key=lambda x: (x[1], x[2]))
    return {
        "category": best[0],
        "label": CATEGORY_LABEL[best[0]],
        "score": best[1],
    }