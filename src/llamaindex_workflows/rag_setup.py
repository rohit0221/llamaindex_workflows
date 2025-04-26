import os
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Load environment variables
load_dotenv()
# rag_setup.py


# --- Load from Environment Variables ---
LLAMA_INDEX_NAME = os.getenv("LLAMA_INDEX_NAME")
LLAMA_PROJECT_NAME = os.getenv("LLAMA_PROJECT_NAME")
LLAMA_ORG_ID = os.getenv("LLAMA_ORG_ID")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")

assert all([LLAMA_INDEX_NAME, LLAMA_PROJECT_NAME, LLAMA_ORG_ID, LLAMA_API_KEY]), "Environment variables missing."

# --- Initialize LlamaCloudIndex and Retriever ---

# Connect to existing index
index = LlamaCloudIndex(
    name=LLAMA_INDEX_NAME,
    project_name=LLAMA_PROJECT_NAME,
    organization_id=LLAMA_ORG_ID,
    api_key=LLAMA_API_KEY,
)

# Create retriever
retriever = index.as_retriever(
    dense_similarity_top_k=6,
    enable_reranking=True,
)

# --- Export retriever ---
def get_rag_retriever():
    return retriever
