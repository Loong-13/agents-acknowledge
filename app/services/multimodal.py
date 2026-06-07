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
        DocType.MARKDOWN.value:1.0,
        DocType.IMAGE.value:0.85,
        DocType.TABLE.value:0.9,
        DocType.PDF.value:0.95,
    }

    def __init__(self):
        self.embeddings = DashScopeEmbeddings(
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )

    async def embed_chunks(self,chunks:list[DocumentChunk])->list[list[float]]:
        """批量嵌入文档快"""
        texts = [chunk.content for chunk in chunks]
        return await self.embeddings.embed_documents(texts)

    async def embed_query(self,query:str)->list[float]:
        """嵌入查询"""
        return await self.embeddings.embed_query(query)

    def weighted_rerank(
            self,
            results:list[tuple[DocumentChunk,float]],
    )->list[MultiModalServiceResult]:
        """
          根据模态权重对结果进行加权排序
          """
        reranked:list[MultiModalServiceResult]=[]
        for chunk,score in results:
            weight=self.MODALITY_WEIGHTS.get(chunk.doc_type.value,1.0)
            reranked.append(
                MultiModalServiceResult(
                    content=chunk.content,
                    modality=chunk.doc_type.value,
                    score=score*weight,
                    metadata=chunk.metadata,
                )
            )
        reranked.sort(key=lambda x:x.score,reverse=True)
        return reranked