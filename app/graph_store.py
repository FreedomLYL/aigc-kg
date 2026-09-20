"""图谱存储层：
- 默认 MemoryGraphStore：零依赖，适合本机快速演示
- 配置了 Neo4j 时改用 Neo4jGraphStore 写真实图库
统一对外接口 save / get_graph 等。
"""
from typing import Dict, List, Optional

from .schemas import ExtractionResult

# 关系类型中文名（供前端/路径展示）
REL_LABEL = {
    "pre": "前置",
    "has_part": "包含",
    "related": "相关",
    "applies_to": "应用",
    "contrast": "对比",
}


class MemoryGraphStore:
    """内存存储：graph_id -> {nodes, links}。"""

    def __init__(self):
        self._graphs: Dict[str, dict] = {}

    def save(self, graph_id: str, extraction: ExtractionResult) -> None:
        nodes = [
            {"id": e.name, "name": e.name, "category": e.kind, "definition": e.definition or ""}
            for e in extraction.entities
        ]
        seen = {n["id"] for n in nodes}
        links = [
            {"source": r.source, "target": r.target, "type": r.type, "label": REL_LABEL.get(r.type, r.type)}
            for r in extraction.relations
            if r.source in seen and r.target in seen
        ]
        self._graphs[graph_id] = {"nodes": nodes, "links": links}

    def get_graph(self, graph_id: str) -> Optional[dict]:
        return self._graphs.get(graph_id)

    def list_graphs(self):
        return list(self._graphs.keys())

    def add_node(self, graph_id: str, name: str, kind: str, definition: str) -> bool:
        g = self._graphs.get(graph_id)
        if not g:
            return False
        if any(n["name"] == name for n in g["nodes"]):
            return False
        g["nodes"].append({"id": name, "name": name, "category": kind, "definition": definition})
        return True

    def update_node(self, graph_id: str, old_name: str, new_name: str = None,
                    kind: str = None, definition: str = None) -> bool:
        g = self._graphs.get(graph_id)
        if not g:
            return False
        node = next((n for n in g["nodes"] if n["name"] == old_name), None)
        if not node:
            return False
        if new_name and new_name != old_name:
            if any(n["name"] == new_name for n in g["nodes"]):
                return False
            node["id"] = new_name
            node["name"] = new_name
            for link in g["links"]:
                if link["source"] == old_name:
                    link["source"] = new_name
                if link["target"] == old_name:
                    link["target"] = new_name
        if kind:
            node["category"] = kind
        if definition is not None:
            node["definition"] = definition
        return True

    def delete_node(self, graph_id: str, name: str) -> bool:
        g = self._graphs.get(graph_id)
        if not g:
            return False
        before = len(g["nodes"])
        g["nodes"] = [n for n in g["nodes"] if n["name"] != name]
        g["links"] = [l for l in g["links"] if l["source"] != name and l["target"] != name]
        return len(g["nodes"]) < before

    def add_edge(self, graph_id: str, source: str, target: str, rel_type: str) -> bool:
        g = self._graphs.get(graph_id)
        if not g:
            return False
        names = {n["name"] for n in g["nodes"]}
        if source not in names or target not in names:
            return False
        exists = any(
            l["source"] == source and l["target"] == target and l["type"] == rel_type
            for l in g["links"]
        )
        if exists:
            return False
        g["links"].append({
            "source": source, "target": target, "type": rel_type,
            "label": REL_LABEL.get(rel_type, rel_type),
        })
        return True

    def delete_edge(self, graph_id: str, source: str, target: str, rel_type: str = None) -> bool:
        g = self._graphs.get(graph_id)
        if not g:
            return False
        before = len(g["links"])
        if rel_type:
            g["links"] = [
                l for l in g["links"]
                if not (l["source"] == source and l["target"] == target and l["type"] == rel_type)
            ]
        else:
            g["links"] = [
                l for l in g["links"]
                if not (l["source"] == source and l["target"] == target)
            ]
        return len(g["links"]) < before

    def delete_course(self, graph_id: str) -> None:
        self._graphs.pop(graph_id, None)


class Neo4jGraphStore:
    """真实图数据库写入。

    使用前需安装：pip install neo4j，并在 .env 配置 NEO4J_URI/USER/PASSWORD。
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # 延迟导入，避免未安装时报错

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def save(self, graph_id: str, extraction: ExtractionResult) -> None:
        with self._driver.session() as session:
            # 清空该课程旧图
            session.run("MATCH (n:Course {id:$gid}) OPTIONAL MATCH (n)-[r*0..2]-() DELETE r, n", gid=graph_id)
            session.run("CREATE (c:Course {id:$gid, name:$gid})", gid=graph_id)
            for e in extraction.entities:
                session.run(
                    "MERGE (k:Knowledge {name:$name}) ON CREATE SET k.kind=$kind, k.definition=$definition",
                    name=e.name, kind=e.kind, definition=e.definition or "",
                )
                session.run(
                    "MATCH (c:Course {id:$gid}), (k:Knowledge {name:$name}) CREATE (c)-[:CONTAINS]->(k)",
                    gid=graph_id, name=e.name,
                )
            for r in extraction.relations:
                session.run(
                    """
                    MATCH (a:Knowledge {name:$src}), (b:Knowledge {name:$tgt})
                    CREATE (a)-[:%s]->(b)
                    """ % r.type,          # type 来自受控枚举，见 schemas 约束
                    src=r.source, tgt=r.target,
                )

    def get_graph(self, graph_id: str) -> Optional[dict]:
        with self._driver.session() as session:
            nodes = session.run(
                "MATCH (c:Course {id:$gid})-[:CONTAINS]->(k:Knowledge) "
                "RETURN k.name AS id, k.name AS name, k.kind AS category, k.definition AS definition",
                gid=graph_id,
            ).data()
            links = session.run(
                "MATCH (c:Course {id:$gid})-[:CONTAINS]->(:Knowledge)-[r]->(:Knowledge)"
                "RETURN startNode(r).name AS source, endNode(r).name AS target, type(r) AS type",
                gid=graph_id,
            ).data()
        mapped = [{"id": n["id"], "name": n["name"], "category": n["category"], "definition": n["definition"] or ""} for n in nodes]
        return {
            "nodes": mapped,
            "links": [{"source": l["source"], "target": l["target"], "type": l["type"], "label": REL_LABEL.get(l["type"], l["type"])} for l in links],
        }

    def list_graphs(self) -> List[str]:
        with self._driver.session() as s:
            return [g["id"] for g in s.run("MATCH (c:Course) RETURN c.id AS id").data()]

    def delete_course(self, graph_id: str) -> None:
        with self._driver.session() as s:
            s.run("MATCH (c:Course {id:$gid}) DETACH DELETE c", gid=graph_id)


class SQLiteGraphStore:
    """默认持久化存储：图谱写入 SQLite（随 app/db.py 一套数据库落盘，重启不丢）。

    编辑逻辑复用上游的 MemoryGraphStore；每次写操作后同步回写 SQLite。
    """

    def __init__(self):
        from . import db
        self._db = db
        self._mem = MemoryGraphStore()

    def _ensure_loaded(self, graph_id: str) -> None:
        if graph_id in self._mem._graphs:
            return
        data = self._db.load_graph(graph_id)
        if data is not None:
            self._mem._graphs[graph_id] = data

    def _persist(self, graph_id: str) -> None:
        if graph_id in self._mem._graphs:
            self._db.save_graph(graph_id, self._mem._graphs[graph_id])

    def save(self, graph_id: str, extraction: ExtractionResult) -> None:
        self._mem.save(graph_id, extraction)
        self._persist(graph_id)

    def get_graph(self, graph_id: str) -> Optional[dict]:
        self._ensure_loaded(graph_id)
        return self._mem.get_graph(graph_id)

    def list_graphs(self) -> List[str]:
        return self._db.list_graphs()

    def delete_course(self, graph_id: str) -> None:
        self._mem._graphs.pop(graph_id, None)
        self._db.delete_course(graph_id)

    def add_node(self, graph_id: str, name: str, kind: str, definition: str) -> bool:
        self._ensure_loaded(graph_id)
        if graph_id not in self._mem._graphs:
            return False
        ok = self._mem.add_node(graph_id, name, kind, definition)
        if ok:
            self._persist(graph_id)
        return ok

    def update_node(self, graph_id: str, old_name: str, new_name: str = None,
                    kind: str = None, definition: str = None) -> bool:
        self._ensure_loaded(graph_id)
        ok = self._mem.update_node(graph_id, old_name, new_name, kind, definition)
        if ok:
            self._persist(graph_id)
        return ok

    def delete_node(self, graph_id: str, name: str) -> bool:
        self._ensure_loaded(graph_id)
        ok = self._mem.delete_node(graph_id, name)
        if ok:
            self._persist(graph_id)
        return ok

    def add_edge(self, graph_id: str, source: str, target: str, rel_type: str) -> bool:
        self._ensure_loaded(graph_id)
        ok = self._mem.add_edge(graph_id, source, target, rel_type)
        if ok:
            self._persist(graph_id)
        return ok

    def delete_edge(self, graph_id: str, source: str, target: str, rel_type: str = None) -> bool:
        self._ensure_loaded(graph_id)
        ok = self._mem.delete_edge(graph_id, source, target, rel_type)
        if ok:
            self._persist(graph_id)
        return ok


_store: Optional[object] = None


def get_store():
    """全局单例存储：有 Neo4j 配置用图库，否则默认 SQLite 持久化。"""
    global _store
    if _store is None:
        from . import config
        if config.use_neo4j():
            _store = Neo4jGraphStore(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
        else:
            _store = SQLiteGraphStore()
    return _store