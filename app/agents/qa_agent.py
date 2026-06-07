from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


INTENT_PROMPT = """
你是一个查询意图分类器。请把用户问题归类为下面五类之一：
- factoid：事实型问题，例如“是谁、是什么、在哪里、什么时候”
- analytical：分析型问题，例如“为什么、如何理解、原因是什么”
- comparative：对比型问题，例如“A 和 B 有什么区别”
- procedural：流程型问题，例如“怎么做、步骤是什么、启动顺序是什么”
- exploratory：探索型问题，例如“有哪些、概览、总结”

只返回意图名称本身，不要返回解释。可选值：factoid、analytical、comparative、procedural、exploratory。
"""

QUERY_REWRITE_PROMPT = """
你是一个企业知识库检索改写助手。请将用户问题改写为更适合向量检索和知识图谱检索的形式。

要求：
1. 生成 1 到 3 个检索查询，尽量保留用户问题中的关键实体和业务术语。
2. 提取问题中明确出现的实体。
3. 提取适合图谱检索的关键词。
4. 只返回 JSON，不要返回 Markdown、解释或额外文本。

返回格式：
{
  "queries": ["检索查询1", "检索查询2"],
  "entities": ["实体1"],
  "keywords": ["关键词1"]
}
"""

ANSWER_PROMPT = """
你是一个企业知识库问答助手。请严格基于提供的上下文回答用户问题。

要求：
1. 只能使用上下文中的信息，不要编造。
2. 如果上下文不足以回答，请明确说明“当前已入库知识不足以回答”。
3. 回答要简洁、准确，优先给出结论，再给出必要依据。
4. 如果问题涉及流程或步骤，请用编号列表回答。
5. 引用信息来源时使用上下文中的 Source 编号或 source 名称。
"""


class QAAgent:
    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    async def answer(self, question: str) -> QAResult:
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        vector_contexts = await self._retrieve_vector(rewritten)
        graph_contexts = await self._retrieve_graph(question, rewritten)

        all_contexts = self._hybrid_rerank(vector_contexts + graph_contexts)
        top_contexts = all_contexts[:8]

        answer_text, reasoning = await self._generate_answer(
            question,
            top_contexts,
            intent,
        )

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
        )

    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        raw = str(resp.content).strip().lower()
        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    async def _rewrite_query(self, question: str) -> dict[str, Any]:
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        data = self._parse_json_object(str(resp.content))
        if not data:
            return {"queries": [question], "entities": [], "keywords": []}

        queries = data.get("queries") or [question]
        if isinstance(queries, str):
            queries = [queries]
        entities = data.get("entities") or []
        keywords = data.get("keywords") or []

        return {
            "queries": [str(q) for q in queries if str(q).strip()] or [question],
            "entities": [str(e) for e in entities if str(e).strip()],
            "keywords": [str(k) for k in keywords if str(k).strip()],
        }

    async def _retrieve_vector(self, rewritten: dict[str, Any]) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in rewritten.get("queries", []):
            results = await self.vector_store.search(query, 5)
            for doc, score in results:
                metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
                contexts.append(
                    RetrievedContext(
                        content=doc.get("content", "") if isinstance(doc, dict) else str(doc),
                        source=metadata.get("source") or doc.get("source", "vector_store"),
                        score=float(score),
                        retrieval_type="vector",
                        metadata=metadata,
                    )
                )
        return contexts

    async def _retrieve_graph(
        self,
        question: str,
        rewritten: dict[str, Any],
    ) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        contexts: list[RetrievedContext] = []
        seen: set[str] = set()
        terms = rewritten.get("entities", []) + rewritten.get("keywords", [])
        if not terms:
            terms = self._fallback_terms(question)

        for term in terms[:6]:
            try:
                records = await self.knowledge_graph.search_entities(term, limit=5)
                for record in records:
                    name = record.get("name", "")
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    neighbors = await self.knowledge_graph.get_neighbors(name, hops=1)
                    contexts.append(
                        RetrievedContext(
                            content=self._format_graph_context(record, neighbors),
                            source="knowledge_graph",
                            score=0.8,
                            retrieval_type="graph",
                            metadata={"entity": name},
                        )
                    )
            except Exception:
                continue
        return contexts

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        weight_map = {
            "vector": 1.0,
            "graph": 1.2,
            "hybrid": 1.1,
        }
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)

        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:160]
            if key not in seen:
                seen.add(key)
                unique.append(ctx)
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        context_text = "\n\n".join(
            f"[Source {i + 1}: {c.source} | type: {c.retrieval_type} | score: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"intent: {intent.value}",
            f"contexts: {len(contexts)}",
            f"vector contexts: {sum(1 for c in contexts if c.retrieval_type == 'vector')}",
            f"graph contexts: {sum(1 for c in contexts if c.retrieval_type == 'graph')}",
        ]

        if not context_text:
            return "没有检索到足够的上下文，无法基于已入库知识回答。", reasoning_steps

        messages = [
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(
                content=f"Context:\n{context_text}\n\nQuestion:\n{question}",
            ),
        ]
        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("answer generated")
        return str(resp.content), reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any] | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        if cleaned.startswith("'''"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("'''", 1)[0]

        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _fallback_terms(question: str) -> list[str]:
        terms = re.findall(r"[\w\u4e00-\u9fff]{2,}", question)
        return terms[:6]

    @staticmethod
    def _format_graph_context(record: dict[str, Any], neighbors: list[dict[str, Any]]) -> str:
        parts = [
            f"Entity: {record.get('name', '')}",
            f"Type: {record.get('type', '')}",
            f"Description: {record.get('description', '')}",
        ]
        if neighbors:
            parts.append("Relations:")
            for item in neighbors[:8]:
                relations = ",".join(item.get("relations", []))
                parts.append(
                    f"- {item.get('source')} -[{relations}]- {item.get('target')}: {item.get('target_desc', '')}"
                )
        return "\n".join(parts)
