"""
LangGraph 编排引擎 — 4 Agent 混合编排

编排模式:
  1. 文档入库流程: DocParser → KnowledgeExtract → (VectorStore + KnowledgeGraph)
  2. 问答流程: Query → QA Agent → (VectorRetrieval ∥ GraphRetrieval) → Answer
  3. 增量更新流程: CDC Event → UpdateAgent → (Diff → Parse → Store)

使用 LangGraph StateGraph 实现有向图编排，支持条件路由和并行分支
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.doc_parser_agent import DocParserAgent, DocumentChunk
from app.agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from app.agents.knowledge_update_agent import (
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
    UpdateResult,
)
from app.agents.qa_agent import QAAgent, QAResult
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.vector_store import VectorStoreService

class workflowType(str,Enum):
    INGEST="ingest"
    QA="qa"
    UPDATE="update"

class IngestState(dict):
    """
    文档入库流程状态
    """
    file_path: list[str]
    chunks: list[DocumentChunk]
    extraction:list[ExtractionResult]
    vector_stored:int
    entities_stored:int
    messages:Annotated[list,add_messages]

class QAState(dict):
    """
    问答流程状态
    """
    question:str
    answer:QAResult| None
    messages:Annotated[list,add_messages]

class UpdateState(dict):
    """
    增量更新流程状态
    """
    changes:list[DocumentChange]
    update_result:UpdateResult
    messages:Annotated[list,add_messages]

def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None=None,
    knowledge_graph: KnowledgeGraphService | None=None,
) -> dict[str, Any]:
    """
    构建三条编排流水线，返回 {"ingest": graph, "qa": graph, "update": graph}
    """
    doc_parser = DocParserAgent()
    extractor=KnowledgeExtractAgent()
    qa_agent=QAAgent(vector_store,knowledge_graph)
    update_agent=KnowledgeUpdateAgent(doc_parser,extractor,vector_store,knowledge_graph)

    return {
        "ingest":_build_ingest_graph(doc_parser,extractor,vector_store,knowledge_graph),
        "qa":_build_qa_graph(qa_agent),
        "update":_build_update_graph(update_agent),
    }

def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
) :
    async def parse_documents(state: dict)-> dict:
        file_paths=state.get("file_paths",[])
        chunks=await doc_parser.parse_batch(file_paths)
        return {"chunks":chunks}

    async def extract_knowledge(state: dict)-> dict:
        chunks=state.get("chunks",[])
        extractions=await extractor.extract(chunks)
        return {"extractions":extractions}
    async def store_vectors(state: dict)-> dict:
        chunks=state.get("chunks",[])
        count=0
        if vector_store and  chunks:
            count+=await vector_store.add_chunks(chunks)
        return {"vector_stored":count}
    async def store_graph(state: dict)-> dict:
        extractions=state.get("extractions",[])
        entity_count=0
        if knowledge_graph:
            for ext in extractions:
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(ent)
                    entity_count+=1
                for rel in ext.relations:
                    await knowledge_graph.add_relation(rel)
        return {"entities_stored":entity_count}
    graph=StateGraph(dict)
    graph.add_node("parse",parse_documents)
    graph.add_node("extract",extract_knowledge)
    graph.add_node("store_vectors",store_vectors)
    graph.add_node("store_graph",store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse","extract")
    graph.add_edge("extract","store_vectors")
    graph.add_edge("store_vectors","store_graph")
    graph.add_edge("store_graph",END)
    graph.add_edge("store_vectors",END)

    return graph.compile()

def _build_qa_graph(qa_agent: QAAgent) :
    async def process_question( state: dict)-> dict:
        question=state.get("question","")
        result=await qa_agent.answer(question)
        return {"answer":result}
    graph=StateGraph(dict)
    graph.add_node("process",process_question)
    graph.set_entry_point("process")
    graph.add_edge("process",END)

    return graph.compile()

def _build_update_graph(update_agent: KnowledgeUpdateAgent) :
    async def process_update(state: dict)-> dict:
        changes=state.get("changes",[])
        results=await update_agent.process_batch(changes)
        return {"results":results}

    def should_connection(state: dict)->str:
        results=state.get("results",[])
        failed=[r for r in results if not r.success]
        if failed:
            return "retry"
        return "done"

    async def retry_failed(state: dict)-> dict:
        results=state.get("results",[])
        failed_changes=[r.change for r in results if not r.success]
        retried=await update_agent.process_batch(failed_changes)
        all_results=[r for r in results if r.success]+retried
        return {"results":all_results}
    graph=StateGraph(dict)
    graph.add_node("process",process_update)
    graph.add_node("retry",retry_failed)
    graph.set_entry_point("process")
    graph.add_conditional_edges("process",should_connection,{"retry":"retry","done":END})
    graph.add_edge("retry",END)

    return graph.compile()
