from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    StartEvent,
    ReflectionEvent,
    RetrievalEvent,
)
from llama_index.llms.openai import OpenAI  # or Gemini if you prefer
import logging
import os
import logging
from dotenv import load_dotenv
# --- Setup LLM client ---
llm = OpenAI(model="gpt-4o-mini")  # Or switch to your preferred model

# --- Kickoff Step (initial start) ---
load_dotenv()

@step(workflow=ResearchWorkflow)
async def kickoff_reflection(ctx: Context, ev: StartEvent) -> ReflectionEvent:
    """
    Kick off by emitting ReflectionEvent directly from StartEvent
    """
    await ctx.set("user_query", ev.query)  # Store original query in context
    logging.info(f"[Kickoff] User query: {ev.query}")

    return ReflectionEvent(query=ev.query)

# --- Reflection Step (real LLM call to refine query) ---

@step(workflow=ResearchWorkflow)
async def reflect_query(ctx: Context, ev: ReflectionEvent) -> RetrievalEvent:
    """
    Use LLM to improve/refine user query for better retrieval.
    """

    logging.info(f"[Reflect] Received query to reflect: {ev.query}")

    system_prompt = (
        "You are a search query expert. Given the user query below, improve it "
        "to maximize relevance and precision for a financial knowledge base."
    )

    full_prompt = f"{system_prompt}\n\nUser Query: {ev.query}\n\nImproved Query:"

    # --- Call LLM ---
    response = await llm.acomplete(full_prompt)

    improved_query = response.text.strip()

    # --- Log improvement ---
    logging.info(f"[Reflect] Improved query: {improved_query}")

    # Store reflected query
    await ctx.set("reflected_query", improved_query)

    return RetrievalEvent(query=improved_query)
