# reflection_steps.py

from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    StartEvent,
    ReflectionEvent,
    RetrievalEvent,
    FinalResponseEvent,
)
from llama_index.llms.openai import OpenAI

import logging
logger = logging.getLogger("app_logger")

# Setup LLM
llm = OpenAI(model="gpt-4o-mini")

# --- Kickoff Step ---
@step(workflow=ResearchWorkflow)
async def kickoff_reflection(ctx: Context, ev: StartEvent) -> ReflectionEvent:
    """
    Kickoff by emitting ReflectionEvent directly from StartEvent
    """
    logger.info("[kickoff_reflection][reflection_steps.py] Entered kickoff_reflection step.")
    logger.info(f"[kickoff_reflection][reflection_steps.py] Received user query: {ev.query}")

    await ctx.set("user_query", ev.query)
    logger.info("[kickoff_reflection][reflection_steps.py] Saved user query into context.")

    return ReflectionEvent(query=ev.query)

# --- Reflection Step ---
@step(workflow=ResearchWorkflow)
async def reflect_query(ctx: Context, ev: ReflectionEvent) -> RetrievalEvent:
    """
    Improve/refine user query for better retrieval.
    """
    logger.info("[reflect_query][reflection_steps.py] Entered reflect_query step.")
    logger.info(f"[reflect_query][reflection_steps.py] Query to reflect: {ev.query}")

    system_prompt = (
        "You are a search query expert. Given the user query below, improve it "
        "to maximize relevance and precision for a financial knowledge base."
    )
    full_prompt = f"{system_prompt}\n\nUser Query: {ev.query}\n\nImproved Query:"

    logger.info("[reflect_query][reflection_steps.py] Calling LLM with reflection prompt.")
    response = await llm.acomplete(full_prompt)
    improved_query = response.text.strip()

    logger.info(f"[reflect_query][reflection_steps.py] Got improved query: {improved_query}")

    await ctx.set("reflected_query", improved_query)
    logger.info("[reflect_query][reflection_steps.py] Saved improved query into context.")

    return RetrievalEvent(query=improved_query)

# --- Finalize Reflection Step ---
@step(workflow=ResearchWorkflow)
async def finalize_reflection(ctx: Context, ev: RetrievalEvent) -> FinalResponseEvent:
    """
    Finalize the reflection-only workflow by emitting the reflected query.
    """
    logger.info("[finalize_reflection][reflection_steps.py] Entered finalize_reflection step.")
    logger.info(f"[finalize_reflection][reflection_steps.py] Reflected query to finalize: {ev.query}")

    return FinalResponseEvent(result=f"Reflected Query: {ev.query}")
