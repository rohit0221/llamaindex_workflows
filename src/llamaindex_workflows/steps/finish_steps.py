from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import JudgmentEvent, FinalResponseEvent

# --- Finalize Answer Step ---

@step(workflow=ResearchWorkflow)
async def finalize_answer(ctx: Context, ev: JudgmentEvent) -> FinalResponseEvent:
    """
    Finalize by emitting FinalResponseEvent with best answer.
    """
    best_answer = ev.responses[0] if ev.responses else "No good answer found."

    return FinalResponseEvent(result=best_answer)
