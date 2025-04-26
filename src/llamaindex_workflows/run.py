# run.py

import asyncio
from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import StartEvent
from llama_index.utils.workflow import draw_most_recent_execution
from llamaindex_workflows.logger import setup_logging


# Setup logging only ONCE and reuse
logger = setup_logging()

async def main():
    # Instantiate the workflow
    workflow = ResearchWorkflow(timeout=300, verbose=True)
    logger.info("[main][run.py] ResearchWorkflow instantiated.")

    # Ask the user for the research question
    user_query = input("\nEnter your research question: ").strip()
    logger.info(f"[main][run.py] User query input: {user_query}")

    # Start the workflow with the user-provided query
    handler = workflow.run(query=user_query)
    logger.info("[main][run.py] Workflow run handler created.")

    # Stream intermediate events as they happen
    print("\nStreaming intermediate events...\n")
    async for event in handler.stream_events():
        print(f"Event Streamed: {event}")
        logger.info(f"[main][run.py] Streamed Event: {event}")

    # Await the final result
    final_result = await handler
    print("\n=== FINAL RESULT ===")
    print(final_result)
    logger.info(f"[main][run.py] Final result received: {final_result}")

    # Draw the agent execution flow
    print("\nGenerating execution flow visualization (most_recent_execution.html)...")
    draw_most_recent_execution(workflow, filename="most_recent_execution.html")
    print("Visualization saved. Open 'most_recent_execution.html' to view the execution path.")
    logger.info("[main][run.py] Execution flow visualization saved to most_recent_execution.html.")

if __name__ == "__main__":
    asyncio.run(main())
