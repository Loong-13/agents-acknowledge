import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.orchestrator.graph import build_knowledge_graph_workflow
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.vector_store import VectorStoreService
from router.qa import router as qa
from router.ingest import router as ingest
from router.admin import router as admin
vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
workflows: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.getenv("UPLOAD_DIR"), exist_ok=True)
    try:
        await vector_store.init()
    except Exception:
        pass
    try:
        await knowledge_graph.init()
    except Exception:
        pass
    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    yield
    await knowledge_graph.close()
app = FastAPI(
    title="AgentKnowledgeHub — 多Agent企业知识管理系统",
    description="支持多模态RAG、知识图谱、增量更新的企业级知识管理 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa)
app.include_router(ingest)
app.include_router(admin)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=os.getenv("API_HOST"), port=int(os.getenv("API_PORT")), reload=True)