import os
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import runtime
from app.schemas.api import IngestResponse

router = APIRouter(prefix="/api/ingest",tags=["文档入库"])
@router.post("/upload",response_model=IngestResponse)
async def upload_document(file:UploadFile=File(...)):
    """上传并解析文档，自动入库到向量库和知识图谱"""
    save_path=os.path.join(os.getenv("UPLOAD_DIR") or "uploads",file.filename or "unknown")
    with open(save_path,"wb") as f:
        shutil.copyfileobj(file.file,f)

    ingest_wf=runtime.workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=500,detail="文档入库工作流不存在")

    result=await ingest_wf.ainvoke({"file_path":save_path})
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

@router.post("/batch",response_model=IngestResponse)
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传并解析文档，自动入库到向量库和知识图谱"""
    results=[]
    for file in files:
        result=await upload_document(file)
        results.append(result)
    return results
