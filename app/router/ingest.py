import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import runtime
from app.schemas.api import IngestResponse

router = APIRouter(prefix="/api/ingest",tags=["文档入库"])
@router.post("/upload",response_model=IngestResponse)
async def upload_document(file:UploadFile=File(...)):
    """上传并解析文档，自动入库到向量库和知识图谱"""
    if not runtime.vector_store_ready:
        raise HTTPException(status_code=503, detail="向量数据库未初始化，请检查 PGVECTOR_DSN 和 pgvector 服务")
    if not runtime.knowledge_graph_ready:
        raise HTTPException(status_code=503, detail="知识图谱未初始化，请检查 NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD 和 neo4j 服务")

    upload_dir = os.getenv("UPLOAD_DIR") or "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = os.path.basename(file.filename or "unknown")
    save_path=os.path.join(upload_dir,safe_name)
    with open(save_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    ingest_wf=runtime.workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=500,detail="文档入库工作流不存在")

    result=await ingest_wf.ainvoke({"file_paths":[save_path]})
    chunks=result.get("chunks",[])
    extractions=result.get("extractions",[])
    total_entities=sum([len(ext.entities) for ext in extractions])
    total_relations=sum([len(ext.relations) for ext in extractions])

    return IngestResponse(
        file_name=file.filename or "unknown",
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success" ,
    )

@router.post("/batch",response_model=list[IngestResponse])
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传并解析文档，自动入库到向量库和知识图谱"""
    results=[]
    for file in files:
        result=await upload_document(file)
        results.append(result)
    return results
