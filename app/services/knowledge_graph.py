"""
知识图谱服务 — Neo4j 图数据库操作

职责:
  1. 实体 (Node) CRUD — 带版本号和时间戳
  2. 关系 (Relationship) CRUD
  3. Cypher 查询执行
  4. 子图检索（多跳遍历）
  5. 按来源删除（支持增量更新）
"""


from __future__ import annotations

import os
import time
from typing import Any

from app.agents.knowledge_extract_agent import Entity, Relation

class KnowledgeGraphService:
    """neo4j 知识图谱服务"""

    def __init__(self):
        self._driver:Any=None

    async def init(self):
        from neo4j import AsyncGraphDatabase
        self._driver=AsyncGraphDatabase.driver(
            os.getenv("neo4j_uri"),
            auth=(os.getenv("neo4j_user"),os.getenv("neo4j_password")),
        )
        await self._ensure_indexes()

    async def close(self)->None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self)->None:
        """创建常用索引以加速查询"""
        index_queries=[
            ""
        ]
