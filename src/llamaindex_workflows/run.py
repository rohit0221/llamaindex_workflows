import asyncio
from llamaindex_workflows.workflow import ResearchWorkflow
from llama_index.core.workflow import StartEvent
from llama_index.utils.workflow import draw_most_recent_execution

async def main():
    # Instantiate the workflow
    workflow = ResearchWorkflow(timeout=60, verbose=True)

    # Ask the user for the research question
    user_query = input("\nEnter your research question: ")

    # Start the workflow with the user-provided query
    handler = workflow.run(query=user_query)

    # Stream intermediate events as they happen
    print("\nStreaming intermediate events...\n")
    async for event in handler.stream_events():
        print(f"Event Streamed: {event}")

    # Await the final result
    final_result = await handler
    print("\n=== FINAL RESULT ===")
    print(final_result)

    # Visualize the execution flow
    print("\nGenerating execution flow visualization (most_recent_execution.html)...")
    draw_most_recent_execution(workflow, filename="most_recent_execution.html")
    print("Visualization saved. Open 'most_recent_execution.html' to view the execution path.")

if __name__ == "__main__":
    asyncio.run(main())
