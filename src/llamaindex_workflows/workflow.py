from llama_index.core.workflow import Workflow

# --- Define Workflow Class First ---
class ResearchWorkflow(Workflow):
    """
    Autonomous Research Agent Workflow
    """

# --- Import Steps AFTER defining ResearchWorkflow ---
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

# --- Attach Steps ---
kickoff_reflection.workflow = ResearchWorkflow
reflect_query.workflow = ResearchWorkflow

send_naive_rag.workflow = ResearchWorkflow
send_topk_rag.workflow = ResearchWorkflow
send_rerank_rag.workflow = ResearchWorkflow

naive_rag.workflow = ResearchWorkflow
topk_rag.workflow = ResearchWorkflow
rerank_rag.workflow = ResearchWorkflow

judge_responses.workflow = ResearchWorkflow
finalize_answer.workflow = ResearchWorkflow
