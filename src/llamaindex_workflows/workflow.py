from llama_index.core.workflow import Workflow

# 1️⃣ Define your workflow class first
class ResearchWorkflow(Workflow):
    """
    Autonomous Research Agent Workflow
    """

# 2️⃣ Then import _all_ of your steps, in the order they’ll be wired up:
from llamaindex_workflows.steps.reflection_steps import kickoff_reflection, reflect_query
from llamaindex_workflows.steps.retrieval_steps import (
    send_naive_rag,
    send_topk_rag,
    send_rerank_rag,
    naive_rag,
    topk_rag,
    rerank_rag,
)
from llamaindex_workflows.steps.judgment_steps import judge_responses
from llamaindex_workflows.steps.synthesize_steps import synthesize_answer
from llamaindex_workflows.steps.finish_steps import finalize_answer

# 3️⃣ The @step decorators in each file already attach them to ResearchWorkflow
#    (via step(workflow=ResearchWorkflow)), so you don’t need to do it again here.
