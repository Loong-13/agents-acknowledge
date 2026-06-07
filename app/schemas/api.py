from typing import Any

from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    source:list[dict[str,Any]]
    reasoning_steps:list[str]

class StatsResponse(BaseModel):
    vector_store:dict[str,Any]
    knowledge_graph:dict[str,Any]

class UpdateRequest(BaseModel):
    file_path: str
    change_type:str="modified"

class UpdateResponse(BaseModel):
    file_path: str
    vectors_added:int
    vectors_deleted:int
    entities_added:int
    relations_added:int
    success:bool
    processing_time:float

class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str