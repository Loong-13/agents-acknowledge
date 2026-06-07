import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from app.orchestrator.graph import build_knowledge_graph_workflow
from app import runtime
from app.router.qa import router as qa
from app.router.ingest import router as ingest
from app.router.admin import router as admin
from app.services.knowledge_graph import KnowledgeGraphService
from app.services.vector_store import VectorStoreService

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.vector_store = VectorStoreService()
    runtime.knowledge_graph = KnowledgeGraphService()
    os.makedirs(os.getenv("UPLOAD_DIR") or "uploads", exist_ok=True)
    try:
        await runtime.vector_store.init()
    except Exception:
        pass
    try:
        await runtime.knowledge_graph.init()
    except Exception:
        pass
    runtime.workflows.update(
        build_knowledge_graph_workflow(
            vector_store=runtime.vector_store,
            knowledge_graph=runtime.knowledge_graph,
        )
    )
    yield
    if runtime.knowledge_graph:
        await runtime.knowledge_graph.close()
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
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST") or "127.0.0.1",
        port=int(os.getenv("API_PORT") or "8000"),
        reload=True,
    )
