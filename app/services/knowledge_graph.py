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

import time
from typing import Any

from app.agents.knowledge_extract_agent import Entity, Relation