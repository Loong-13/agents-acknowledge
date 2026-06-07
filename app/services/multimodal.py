"""
多模态服务 — 统一处理不同模态数据的嵌入与检索

职责:
  1. 文本嵌入
  2. 图像嵌入（通过 LLM 视觉描述后再嵌入）
  3. 表格嵌入（结构化 → 自然语言 → 嵌入）
  4. 跨模态检索时的分数加权融合
"""

from __future__ import annotations
from typing import Any
import os
from dataclasses import dataclass
from langchain_community.embeddings import DashScopeEmbeddings
from app.agents.doc_parser_agent import DocumentChunk,DocType

@dataclass
class MultiModalServiceResult:
    content:str
    modality:str
    score:float
    metadata:dict[str,Any]

class MultiModalService:
    """
      多模态处理服务

      策略: 各模态先转为文本表示，再做统一嵌入
      不同模态在检索时根据与查询的匹配度施加不同权重
      """
    MODALITY_WEIGHTS:dict[str,float]={
        DocType.TEXT.value:1.0,
        DocType.MARKDOWN.value:1.0
    }