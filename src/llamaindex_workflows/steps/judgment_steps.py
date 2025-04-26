from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    RAGResponseEvent,
    JudgmentEvent,
    ReflectionEvent,
    FinalResponseEvent
)

@step(workflow=ResearchWorkflow)
async def judge_responses(ctx: Context, ev: RAGResponseEvent) -> JudgmentEvent | ReflectionEvent:
    """
    Collect multiple RAG responses, pick best, or loop back if none good.
    """
    collected = ctx.collect_events(ev, [RAGResponseEvent] * 3)
    if collected is None:
        # We have not collected all three yet
        return None

    # Now we have all 3
    responses = [(r.strategy, r.content) for r in collected]

    # Very simple scoring: pick the longest content (simulate best answer)
    best_response = max(responses, key=lambda x: len(x[1]))

    # Threshold to reject if too small
    if len(best_response[1]) < 200:
        # --- Fetch the original user query properly ---
        user_query = await ctx.get("user_query", default="Unknown Query")

        # Re-emit a ReflectionEvent on original query
        return ReflectionEvent(query=user_query)

    # Otherwise return JudgmentEvent with best answers
    return JudgmentEvent(responses=[r[1] for r in responses])
