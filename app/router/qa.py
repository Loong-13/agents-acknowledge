from fastapi import APIRouter, HTTPException

from app import runtime
from app.schemas.api import QuestionRequest, QuestionResponse

router=APIRouter(prefix="/api/qa",tags=["智能回答"])

@router.post("/ask",response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + 知识图谱推理"""
    qa_wf=runtime.workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503,detail="QA workflow not initialized")

    result=await qa_wf.ainvoke({"question":req.question})
    qa_result=result.get("answer")
    if not qa_result:
        raise HTTPException(status_code=500,detail="QA workflow failed")

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent,
        source=[
            {"content":c.content[:200],"source":c.source,"score":c.score,"type":c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )
