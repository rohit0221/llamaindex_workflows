from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step, Context
from llamaindex_workflows.events import (
    RAGResponseEvent,
    JudgmentEvent,
)

from llama_index.llms.openai import OpenAI  # or Gemini etc
import logging

# --- Setup LLM client ---
llm = OpenAI(model="gpt-4o-mini")

@step(workflow=ResearchWorkflow)
async def synthesize_answer(ctx: Context, ev: JudgmentEvent) -> JudgmentEvent:
    """
    Synthesize multiple retrieved RAG responses into a final better answer.
    """

    responses = ev.responses  # list of text chunks from RAG

    if not responses:
        logging.warning("[Synthesize] No responses to synthesize.")
        return JudgmentEvent(responses=["No useful information retrieved."])

    # Create an LLM prompt
    prompt = (
        "You are a financial report expert.\n\n"
        "Given the following pieces of context from financial reports, synthesize a comprehensive and concise answer:\n\n"
    )
    for idx, resp in enumerate(responses):
        prompt += f"Context {idx+1}:\n{resp}\n\n"
    prompt += "Provide a single clear, complete answer based on the contexts above."

    # --- Call the LLM ---
    response = await llm.acomplete(prompt)

    synthesized_answer = response.text.strip()

    logging.info(f"[Synthesize] Synthesized answer length: {len(synthesized_answer)} chars.")

    # Return as a JudgmentEvent again
    return JudgmentEvent(responses=[synthesized_answer])
