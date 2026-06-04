"""
GraphRAG 混合检索管道 — 向量检索 + 图谱遍历 + 重排序


工作流:
  Query → [向量检索分支] ────→ 合并 → 交叉重排序 → Top-K
         [图谱检索分支] ────↗

图谱检索策略:
  1. 实体链接: 从 query 中识别实体 → 在图谱中定位
  2. 子图召回: 从定位实体出发 N 跳遍历
  3. 路径推理: 找到实体间的最短路径，提供推理链
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService