from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import StartEvent  # Import StartEvent

import os
from dotenv import load_dotenv

from llamaindex_workflows.events import ReflectionEvent, RetrievalEvent

# Load environment variables
load_dotenv()

@step(workflow=ResearchWorkflow)
async def kickoff_reflection(ctx: Context, ev: StartEvent) -> ReflectionEvent:
    """
    Kick off the workflow from StartEvent to ReflectionEvent.
    """
    return ReflectionEvent(query=ev.query)

# --- Reflection Step ---
@step(workflow=ResearchWorkflow)
async def reflect_query(ctx: Context, ev: ReflectionEvent) -> ReflectionEvent | RetrievalEvent:
    """
    Reflect on the quality of the query.
    If bad, improve and emit another ReflectionEvent (loop).
    If good, proceed to RetrievalEvent.
    """
    # Initialize LLM
    llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = f"""You are an expert query evaluator.
Given the following research query, judge if it is well-phrased and specific.
If it is good, respond with 'GOOD'.
If it is vague, suggest an improved version.

Query: {ev.query}
"""
    response = await llm.acomplete(prompt)
    output = response.text.strip()

    if "GOOD" in output.upper():
        return RetrievalEvent(query=ev.query)
    else:
        improved_query = output
        return ReflectionEvent(query=improved_query)
