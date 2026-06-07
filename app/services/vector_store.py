"""
向量存储服务 — 支持 ChromaDB / PGVector 双后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""
from __future__ import annotations
import os
from typing import Any
from langchain_community.embeddings import DashScopeEmbeddings
from app.agents.doc_parser_agent import DocumentChunk

class VectorStoreService:
    """向量存储服务,底层可切换 chromaDB/PGVector"""

    COLLECTION_NAME="knowledge_chunks"

    def __init__(self)->None:
        self.embeddings=DashScopeEmbeddings(
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self._store:Any=None
        self._backend=os.getenv("VECTOR_STORE_TYPE")

    async def init(self)->None:

        await self._init_pgvector()

    async def _init_pgvector(self)->None:
        from langchain_community.vectorstores import PGVector
        self._store=PGVector(
            collection_name=self.COLLECTION_NAME,
            connection_string=os.getenv("PGVECTOR_DSN"),
            embedding_function=self.embeddings,
        )

    async def add_chunks(self,chunks:list[DocumentChunk])->int:
        """向量化并存储文档快"""
        if not chunks:
            return 0

        texts=[c.content for c in chunks]
        ids=[c.chunk_id for c in chunks]
        metadatas=[
            {"doc_id":c.doc_id,"doc_type":c.doc_type.value,"source":c.metadata.get("source",""),"chunk_id":c.chunk_id}
            for c in chunks
        ]

        if hasattr(self._store,"aadd_texts"):
            await self._store.aadd_texts(texts=texts,metadatas=metadatas,ids=ids)
        else:
            self._store.add_texts(texts=texts,metadatas=metadatas,ids=ids)

        return len(chunks)

    async def search(self,query:str,top_k:int=5)->list[tuple[dict, float]]:
        """语义搜索，返回（文档，分数）列表"""
        if hasattr(self._store,"asimilarity_search_with_score"):
            results=await self._store.asimilarity_search_with_score(query,top_k)
        else:
            results=self._store.similarity_search_with_score(query,top_k)
        return [
            ({"content":doc.page_content,"source":doc.metadata.get("source",""),"metadata":doc.metadata},score)
            for doc,score in results
        ]

    async def delete_by_doc_id(self,doc_id:str)->int:
        """按 doc_id 删除文档块"""
        existing=self._store.get(where={"doc_id":doc_id},include=[])
        if not existing:
            return 0
        return await self._store.delete(doc_ids=[doc_id])

    async def get_stats(self)->dict:
        """获取向量数据库统计信息"""
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
