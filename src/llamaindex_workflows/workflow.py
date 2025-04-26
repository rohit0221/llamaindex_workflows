from llama_index.core.workflow import Workflow

# --- Define Workflow Class First ---
class ResearchWorkflow(Workflow):
    """
    Autonomous Research Agent Workflow
    """

# --- Now import all steps ---
from llamaindex_workflows.steps.reflection_steps import (
    kickoff_reflection,
    reflect_query,
)
from llamaindex_workflows.steps.retrieval_steps import (
    send_naive_rag,
    send_topk_rag,
    send_rerank_rag,
    naive_rag,
    topk_rag,
    rerank_rag,
)
from llamaindex_workflows.steps.judgment_steps import judge_responses
from llamaindex_workflows.steps.finish_steps import finalize_answer
