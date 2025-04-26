from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import RAGResponseEvent
from llama_index.core.schema import NodeWithScore
import logging

@step(workflow=ResearchWorkflow)
async def judge_responses(ctx: Context, ev: RAGResponseEvent) -> list[NodeWithScore]:
    """
    Judge and select the best RAG responses (nodes) based on internal score.
    """

    if not ev.responses:
        logging.warning("[Judgment] No RAG responses to judge.")
        return []

    # Simple scoring: take top N nodes
    sorted_nodes = sorted(ev.responses, key=lambda node: node.score or 0.0, reverse=True)
    top_nodes = sorted_nodes[:5]  # Pick top 5 best-scored nodes

    logging.info(f"[Judgment] Selected top {len(top_nodes)} nodes after judging.")
    return top_nodes
