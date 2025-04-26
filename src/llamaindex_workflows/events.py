from llama_index.core.workflow import Event, StopEvent, StartEvent

# --- Events for Query Reflection ---
class ReflectionEvent(Event):
    query: str

# --- Events for Retrieval Phase ---
class RetrievalEvent(Event):
    query: str

class NaiveRAGEvent(Event):
    query: str

class TopKRAGEvent(Event):
    query: str

class RerankRAGEvent(Event):
    query: str

# --- Event for Receiving RAG Responses ---
class RAGResponseEvent(Event):
    content: str
    strategy: str

# --- Event for Judging RAG Outputs ---
class JudgmentEvent(Event):
    responses: list[str]

# --- Final Response Event ---
class FinalResponseEvent(StopEvent):
    result: str
