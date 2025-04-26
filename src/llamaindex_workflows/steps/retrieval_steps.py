from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    RetrievalEvent,
    NaiveRAGEvent,
    TopKRAGEvent,
    RerankRAGEvent,
    RAGResponseEvent
)
from llamaindex_workflows.rag_setup import get_rag_query_engine

# --- Setup retriever once ---
query_engine = get_rag_query_engine()

# --- Split Retrieval Event Launchers ---

@step(workflow=ResearchWorkflow)
async def send_naive_rag(ctx: Context, ev: RetrievalEvent) -> NaiveRAGEvent:
    """
    Launch Naive RAG.
    """
    return NaiveRAGEvent(query=ev.query)

@step(workflow=ResearchWorkflow)
async def send_topk_rag(ctx: Context, ev: RetrievalEvent) -> TopKRAGEvent:
    """
    Launch Top-K RAG.
    """
    return TopKRAGEvent(query=ev.query)

@step(workflow=ResearchWorkflow)
async def send_rerank_rag(ctx: Context, ev: RetrievalEvent) -> RerankRAGEvent:
    """
    Launch Rerank RAG.
    """
    return RerankRAGEvent(query=ev.query)

# --- Retrieval Workers ---

@step(workflow=ResearchWorkflow)
async def naive_rag(ctx: Context, ev: NaiveRAGEvent) -> RAGResponseEvent:
    """
    Naive RAG: simple retrieval.
    """
    response = await query_engine.aquery(ev.query)
    return RAGResponseEvent(
        content=response.response,
        strategy="naive"
    )

@step(workflow=ResearchWorkflow)
async def topk_rag(ctx: Context, ev: TopKRAGEvent) -> RAGResponseEvent:
    """
    Top-K RAG: broader retrieval (simulated).
    """
    response = await query_engine.aquery(ev.query)
    return RAGResponseEvent(
        content=response.response,
        strategy="topk"
    )

@step(workflow=ResearchWorkflow)
async def rerank_rag(ctx: Context, ev: RerankRAGEvent) -> RAGResponseEvent:
    """
    Rerank RAG: re-ranked retrieval (simulated).
    """
    response = await query_engine.aquery(ev.query)
    return RAGResponseEvent(
        content=response.response,
        strategy="rerank"
    )
