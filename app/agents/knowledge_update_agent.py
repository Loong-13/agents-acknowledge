"""
知识更新 Agent — 监听文档变更，增量更新向量库和知识图谱

核心能力:
  1. 文件系统监听 (Watchdog) / Kafka CDC 消费
  2. 差量对比：对比新旧文档，只处理变更部分
  3. 增量向量化 & 图谱更新
  4. 版本管理：知识节点带时间戳和版本号
"""
from __future__ import annotations
import os
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any




@dataclass
class DocumentChange:
    """文档变更信息"""
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str=""
    new_hash: str=""
    diff_chunks:list[str]=field(default_factory=list)

class ChangeType(Enum):
    """文档变更类型"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"

@dataclass
class UpdateResult:
    """更新结果"""
    change:DocumentChange
    vectors_added:int=0
    vectors_updated:int=0
    entities_added:int=0
    entities_updated:int=0
    relations_added:int=0
    relations_updated:int=0
    success:bool=True
    error:str=""
    processing_time:float=0

class KnowledgeUpdateAgent:
    """
      知识更新 Agent

      支持两种模式:
        1. 文件监听模式 (Watchdog): 监听本地文件系统变更
        2. CDC 模式 (Kafka): 消费来自消息队列的变更事件

      工作流:
        detect_change → diff_analysis → incremental_parse → update_vector_store → update_knowledge_graph → log
      """
    def __init__(
        self,
        doc_parser:Any=None,
        knowledge_extractor:Any=None,
        vector_store:Any=None,
        knowledge_graph:Any=None,
    )->None:
        self.doc_parser=doc_parser
        self.knowledge_extractor=knowledge_extractor
        self.vector_store=vector_store
        self.knowledge_graph=knowledge_graph
        self._file_hashes:dict[str, str]= {}
        self._version_counter:dict[str, int]= {}


    async def process_change(self, change:DocumentChange)->UpdateResult:
        """处理文档变更"""
        start_time=time.time()
        result=UpdateResult(change=change)

        try:
            if change.change_type==ChangeType.DELETED:
                await self._handle_delete( change, result)
            elif change.change_type==ChangeType.CREATED:
                await self._handle_create( change, result)
            elif change.change_type==ChangeType.MODIFIED:
                await self._handle_modify( change, result)
        except Exception as e:
            result.success=False
            result.error=str(e)

        result.processing_time=(time.time()-start_time)*1000
        return result

    async def process_batch(self, changes:list[DocumentChange])->list[UpdateResult]:
        """处理批量变更"""
        results:list[UpdateResult]=[]
        for change in changes:
            result=await self.process_change(change)
            results.append(result)
        return  results

    def detect_change(self, file_path:list[str])-> list[DocumentChange]:
        """检测文档变更"""
        changes:list[DocumentChange]=[]
        current_file=set(file_path)

        for fp in current_file:
            old_hash=self._file_hashes.get(fp,"")
            new_hash=self._compute_hash(fp)
            if old_hash != new_hash:
                change=DocumentChange(file_path=fp, change_type=ChangeType.CREATED, new_hash=new_hash,)
                changes.append(change)
            elif new_hash !=old_hash:
                change=DocumentChange(file_path=fp, change_type=ChangeType.UPDATED, old_hash=old_hash, new_hash=new_hash)
                changes.append(change)
                self._file_hashes[fp]=new_hash

        for fp in set(self._file_hashes) - current_file:
            change=DocumentChange(file_path=fp, change_type=ChangeType.DELETED,old_hash=self._file_hashes[fp],)
            changes.append(change)
            del self._file_hashes[fp]

        return changes

    async def start_watchdog(self, directory:str)->None:
        """启动文件监听"""
        import threading
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        agent=self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                import asyncio
                change=DocumentChange(file_path=event.src_path, change_type=ChangeType.CREATED)
                asyncio.run(agent.process_change(change))

            def on_modified(self, event) :
                import asyncio
                change=DocumentChange(file_path=event.src_path, change_type=ChangeType.MODIFIED)
                asyncio.run(agent.process_change(change))

        observer=Observer()
        observer.schedule(_Handler(), directory, recursive=True)

        def _run():
            observer.start()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

        t=threading.Thread(target=_run,daemon= True)
        t.start()

    async def start_kafka_consumer(self)->None:
        """启动 Kafka 消费者"""
        import json
        from confluent_kafka import Consumer
        conf={
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            "group.id":"knowledge-update-agent",
            "auto.offset.reset": "earliest",
        }
        consumer=Consumer(conf)
        consumer.subscribe([os.getenv("KAFKA_TOPIC")])

        try:
            while True:
                msg=consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():

                    continue
                payload=json.loads(msg.value().decode("utf-8"))
                change=DocumentChange(
                    file_path=payload["file_path"],
                    change_type=ChangeType[payload["change_type"]],
                    old_hash=payload["old_hash",""],
                    new_hash=payload["new_hash",""],
                )
                await self.process_change(change)
        finally:
            consumer.close()

    async def _handle_create(self, change:DocumentChange, result:UpdateResult)->None:
        if not self.doc_parser:
            return
        chunks=await self.doc_parser.parse_file(change.file_path)

        if self.vector_store:
            await self.vector_store.delete_chunks(chunks)
            result.vectors_added=len(chunks)

        if self.knowledge_extractor and self.knowledge_graph:
            extractions=await self.knowledge_extractor.extract(chunks)
            for ext in extractions:
                for ent in ext.entities:
                    version =self._bump_version(ent.name)
                    await self.knowledge_graph.upsert_entity(ent.name, version)
                    result.entities_added+=1
                for rel in ext.relations:
                    await self.knowledge_graph.add_relation(rel)
                    result.relations_added+=1

    async def _handle_modify(self,change:DocumentChange, result:UpdateResult):
        doc_id=hashlib.sha256(change.file_path.encode().hexdigest()[:16])

        if self.vector_store:
            deleted=await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted=deleted

        await self._handle_create(change, result)
        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)


    async def _handle_delete(self, change:DocumentChange, result:UpdateResult):
        doc_id=hashlib.sha256(change.file_path.encode().hexdigest()[:16])
        if self.vector_store:
            deleted=await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted=deleted

        if self.knowledge_graph:
            await self.knowledge_graph.delete_by_source(change.file_path)

    @staticmethod
    def _compute_hash(file_path:str)->str:
        """计算文件的 hash"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _bump_version(self, entity_name:str)->int:
        ver =self._version_counter.get(entity_name, 0)+1
        self._version_counter[entity_name]=ver
        return ver

