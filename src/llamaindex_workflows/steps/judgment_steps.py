# judgment_steps.py

from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    RAGResponseEvent,
    JudgmentEvent,
    ReflectionEvent,
)

@step(workflow=ResearchWorkflow)
async def judge_responses(ctx: Context, ev: RAGResponseEvent) -> JudgmentEvent:
    collected = ctx.collect_events(ev, [RAGResponseEvent] * 3)
    if collected is None:
        return None

    responses = [(r.strategy, r.content) for r in collected]
    # Sort by length just to simulate quality
    responses = sorted(responses, key=lambda x: len(x[1]), reverse=True)
    return JudgmentEvent(responses=[r[1] for r in responses])

