from llama_index.utils.workflow import draw_all_possible_flows, draw_most_recent_execution

def visualize_entire_flow(workflow_class, filename="full_flow.html"):
    """
    Visualize all possible flows for a workflow class.
    """
    draw_all_possible_flows(workflow_class, filename=filename)

def visualize_last_run(workflow_instance, filename="most_recent_execution.html"):
    """
    Visualize only the actual last execution path taken by the workflow.
    """
    draw_most_recent_execution(workflow_instance, filename=filename)
