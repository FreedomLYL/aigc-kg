"""学习路径推荐：基于知识图谱的"前置关系"，用简单图遍历实现。

规则：
1. 候选 = 尚未掌握的知识点
2. 一个候选"可学"（ready）的充分条件：指向它的所有前置边(pre)的源都已掌握；
   没有任何前置边的根节点天然 ready
3. 排序：薄弱点(weak，答题答错的知识点) > ready 的核心点 > 其它候选，
   其中 ready 与 in-degree 越大的越靠前
"""

from typing import List, Optional


def recommend(
    graph: Optional[dict],
    mastered: List[str],
    weak: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[dict]:
    if not graph:
        return []
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    name_to_node = {n["name"]: n for n in nodes}

    # 前置边：只把 type 为 pre 的关系当作学习前置
    pre_targets: dict[str, List[str]] = {}
    in_degree: dict[str, int] = {n["name"]: 0 for n in nodes}
    for link in links:
        src, tgt = link["source"], link["target"]
        if src not in name_to_node or tgt not in name_to_node:
            continue
        in_degree[tgt] = in_degree.get(tgt, 0) + 1
        if link.get("type") == "pre":
            pre_targets.setdefault(tgt, []).append(src)

    learned = set(mastered or [])
    weak_set = set(weak or [])
    # 薄弱点若已掌握，仍应优先复习，故单独归入权重最高档
    candidates = [n for n in nodes if n["name"] not in learned or n["name"] in weak_set]

    def ready_score(node_name: str) -> int:
        pereq = pre_targets.get(node_name, [])
        if not pereq:
            return 1  # 无前置 → 可学
        return 1 if all(p in learned for p in pereq) else 0

    def sort_key(n):
        nm = n["name"]
        return (
            2 if nm in weak_set else 0,            # 薄弱点优先复习
            ready_score(nm),                        # 其次可学
            in_degree.get(nm, 0),                   # 再次核心程度
        )

    ranked = sorted(candidates, key=sort_key, reverse=True)
    return [
        {
            "name": n["name"],
            "kind": n["category"],
            "definition": n["definition"],
            "in_degree": in_degree.get(n["name"], 0),
            "weak": n["name"] in weak_set,
        }
        for n in ranked[:top_k]
    ]