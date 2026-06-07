"""
CDC (Change Data Capture) 增量处理器

增量更新流程:
  变更事件 → 差量分析 → 增量解析 → 增量向量化 → 增量图谱更新
              ↓
          版本管理（每个知识节点带 version + timestamp）
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any
import os

@dataclass
class CDCEvent:
    """统一的CDC事件格式"""
    event_id:str
    source_type:str #"filesystem"|"database"|"api"
    operation:str #"Insert"|"Update"|"Delete"
    resource_path:str
    timestamp:float=field(default_factory= time.time)
    before:dict[str,Any] | None = None
    after:dict[str,Any] | None = None
    diff:dict[str,Any] | None = None

@dataclass
class CDCProcessorResult:
    event:CDCEvent
    chunks_affected: int=0
    entities_affected: int=0
    processing_time: float=0
    version:int=0
    success:bool=True
    error:str=""

class CDCProcessor:
    """
        CDC 增量处理器

        核心设计:
          1. 事件归一化: 将不同来源的变更事件统一为 CDCEvent 格式
          2. 差量计算: 对比 before/after，只处理实际变更的内容
          3. 增量处理: 只重新解析、向量化、图谱化变更部分
          4. 版本追踪: 每次更新递增版本号，支持回滚
        """
    def __init__(self):
        self._version_map:dict[str,int] = {}
        self._event_log:list[CDCEvent] = []
        self._processing_queue:list[CDCEvent] = []


    @staticmethod
    def from_filesystem_event(event_type:str,file_path:str,content_before:str="",content_after:str="")-> CDCEvent:
        """从文件系统变更事件生成CDCEvent"""
        op_map={"created":"Insert","modified":"Update","deleted":"Delete"}
        return CDCEvent(
            event_id=hashlib.sha256(f"{file_path}:{time.time()}".encode()).hexdigest()[:16],
            source_type="filesystem",
            operation=op_map.get(event_type,"Update"),
            resource_path=file_path,
            before=json.loads(content_before) if content_before else None,
            after=json.loads(content_after) if content_after else None,

        )

    @staticmethod
    def from_kafka_message(message:bytes)-> CDCEvent:
        """从Kafka消息生成CDCEvent"""
        payload=json.loads(message)
        return CDCEvent(
            event_id=payload.get("id",hashlib.sha256(message).hexdigest()[:16]),
            source_type="database",
            operation=payload.get("operation","Update").upper(),
            resource_path=payload.get("source",{}).get("table","unknown"),
            before=payload.get("before"),
            after=payload.get("after"),
            timestamp=payload.get("ts_ms",time.time()*1000)/1000,
        )

    @staticmethod
    def compute_diff(before:str,after:str)-> dict[str,Any]:
        """计算变更前后的差量"""
        before_lines=before.splitlines() if before else []
        after_lines=after.splitlines() if after else []

        before_set=set(before_lines)
        after_set=set(after_lines)

        added=after_set - before_set
        removed=before_set - after_set

        change_ratio=len(added| removed)/max(len(before_lines)+len(after_lines),1)

        return {
            "added_lines":list(added),
            "removed_lines":list(removed),
            "added_count":len(added),
            "removed_count":len(removed),
            "change_ratio":round(change_ratio,4),
            "is_major_change":change_ratio>0.3,
        }

    def dump_version(self,resource_path:str)-> int:
        """递增资源版本号"""
        current=self._version_map.get(resource_path,0)
        new_version=current+1
        self._version_map[resource_path]=new_version
        return new_version

    def get_version(self,resource_path:str)-> int:
        """获取资源版本号"""
        return self._version_map.get(resource_path,0)

    async def process_event(self,event:CDCEvent)-> CDCProcessorResult:
        """处理一个变更事件"""
        start=time.time()
        result=CDCProcessorResult(event=event)
        try:
            version=self.dump_version(event.resource_path)
            result.version=version

            if event.operation=="Delete":
                result.chunks_affected=-1
                result.entities_affected=-1
            elif event.operation=="Insert":
                result.chunks_affected=1
                result.entities_affected=1
            elif event.operation=="Update":
                if event.before and event.after:
                    diff =self.compute_diff(
                        event.before.get("content",""),
                        event.after.get("content",""),
                    )
                    event.diff=diff
                    if diff["is_major_change"]:
                        result.chunks_affected=diff["added_count"]
                    else:
                        result.chunks_affected=max(1,diff["added_count"]//10)

            self._envent_log.append(event)
        except Exception as e:
            result.success=False
            result.error=str(e)

        result.processing_time=(time.time()-start)*1000
        return result

    async def process_batch(self, events:list[CDCEvent])-> list[CDCProcessorResult]:
        """处理批量变更"""
        results:list[CDCProcessorResult]=[]
        for event in events:
            results.append(await self.process_event(event))
        return results

    async def start_kafka_consumer(self,topics:list[str] |None= None)-> None:
        """启动Kafka消费者"""
        from confluent_kafka import Consumer
        if topics is None:
            topics=[os.getenv("KAFKA_TOPIC_DOC_CHANGES")]

        conf ={
            "bootstrap.servers":os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            "group.id":"cdc-processor",
            "auto.offset.reset":"latest",
            "enable.auto.commit":True,
        }
        consumer=Consumer(conf)
        consumer.subscribe(topics)

        try:
            while True:
                msg=consumer.poll(timeout=1.0)
                if msg is None or msg.error():
                    continue
                event=self.from_kafka_message(msg.value())
                await self.process_event(event)
        finally:
            consumer.close()

    def get_stats(self)-> dict[str,Any]:
        """获取统计信息"""
        return {
            "total_events":len(self._event_log),
            "tracked_resources":len(self._version_map),
            "queue_size":len(self._processing_queue),
            "version":dict(self._version_map),
        }

    def get_event_history(self,resource_path:str|None=None,limit:int=50)-> list[CDCEvent]:
        events=self._event_log
        if resource_path:
            events=[e for e in events if e.resource_path==resource_path]
        return events[-limit:]
