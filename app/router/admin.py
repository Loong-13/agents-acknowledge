from fastapi import APIRouter,HTTPException

from app.agents.knowledge_update_agent import DocumentChange, ChangeType
from app.main import workflows, vector_store, knowledge_graph
from app.schemas.api import StatsResponse, UpdateResponse, UpdateRequest


router = APIRouter(prefix="api/admin",tags=["系统管理"])

@router.get("/stats",response_model=StatsResponse)
async def get_stats():
    """系统统计"""
    vs_stats=await vector_store.get_stats()
    kg_stats=await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats,knowledge_graph=kg_stats)

@router.post("/update",response_model=UpdateResponse)
async def trigger_update(req: UpdateRequest):
    """手动触发知识更新"""
    update_wf=workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=500,detail="知识更新工作流不存在")

    change=DocumentChange(
        file_path=req.file_path,
        change_type=ChangeType(req.change_type),
    )
    result=await update_wf.ainvoke(change)
    results=result.get("results",[])
    if not results:
        raise HTTPException(status_code=500,detail="知识更新失败")

    r=results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time=r.processing_time,
    )

