# retrieval_steps.py

from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    RetrievalEvent,
    NaiveRAGEvent,
    TopKRAGEvent,
    RerankRAGEvent,
    RAGResponseEvent,
)
from llamaindex_workflows.rag_setup import get_rag_retriever

# --- Setup retriever once ---
retriever = get_rag_retriever()

# --- Dispatchers ---

@step(workflow=ResearchWorkflow)
async def send_naive_rag(ctx: Context, ev: RetrievalEvent) -> NaiveRAGEvent:
    """Launch Naive RAG."""
    return NaiveRAGEvent(query=ev.query)

@step(workflow=ResearchWorkflow)
async def send_topk_rag(ctx: Context, ev: RetrievalEvent) -> TopKRAGEvent:
    """Launch Top-K RAG."""
    return TopKRAGEvent(query=ev.query)

@step(workflow=ResearchWorkflow)
async def send_rerank_rag(ctx: Context, ev: RetrievalEvent) -> RerankRAGEvent:
    """Launch Rerank RAG."""
    return RerankRAGEvent(query=ev.query)

# --- Real Retrieval Workers ---

@step(workflow=ResearchWorkflow)
async def naive_rag(ctx: Context, ev: NaiveRAGEvent) -> RAGResponseEvent:
    """Naive RAG retrieval."""
    nodes = await retriever.aretrieve(ev.query)
    combined_content = "\n\n".join(node.get_content() for node in nodes)

    return RAGResponseEvent(
        content=combined_content,
        strategy="naive",
    )

@step(workflow=ResearchWorkflow)
async def topk_rag(ctx: Context, ev: TopKRAGEvent) -> RAGResponseEvent:
    """Top-K RAG retrieval."""
    nodes = await retriever.aretrieve(ev.query)
    combined_content = "\n\n".join(node.get_content() for node in nodes)

    return RAGResponseEvent(
        content=combined_content,
        strategy="topk",
    )

@step(workflow=ResearchWorkflow)
async def rerank_rag(ctx: Context, ev: RerankRAGEvent) -> RAGResponseEvent:
    """Rerank RAG retrieval."""
    nodes = await retriever.aretrieve(ev.query)
    combined_content = "\n\n".join(node.get_content() for node in nodes)

    return RAGResponseEvent(
        content=combined_content,
        strategy="rerank",
    )
