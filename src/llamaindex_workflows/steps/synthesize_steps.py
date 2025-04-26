from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import step
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
import logging

# Setup LLM
llm = OpenAI(model="gpt-4o-mini")

@step(workflow=ResearchWorkflow)
async def synthesize_answer(query: str, nodes: list[NodeWithScore]) -> str:
    """
    Synthesizes a final answer from the user query and selected retrieved nodes.
    """

    if not nodes:
        logging.warning("[Synthesize] No nodes retrieved to synthesize.")
        return "No useful information retrieved."

    # Build context from nodes
    context = "\n\n".join([node.get_content() for node in nodes])

    # Create prompt
    prompt = f"""You are a financial research expert.

Using only the following extracted context, answer the user question as precisely and concisely as possible.

Context:
{context}

Question:
{query}

Answer:"""

    # LLM call
    response = await llm.acall(prompt)
    synthesized_answer = response.text.strip()

    logging.info(f"[Synthesize] Synthesized answer length: {len(synthesized_answer)} characters.")

    return synthesized_answer
