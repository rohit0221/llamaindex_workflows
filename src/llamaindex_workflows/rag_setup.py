import os
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Load environment variables
load_dotenv()

def get_rag_query_engine():
    """
    Returns a real query engine connected to LlamaCloud using environment variables.
    """
    index = LlamaCloudIndex(
        name=os.getenv("LLAMA_INDEX_NAME"),
        project_name=os.getenv("LLAMA_PROJECT_NAME"),
        organization_id=os.getenv("LLAMA_ORG_ID"),
        api_key=os.getenv("LLAMA_API_KEY"),
    )
    return index.as_query_engine()
