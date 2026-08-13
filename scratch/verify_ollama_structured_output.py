import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VECTORSTORE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, OLLAMA_MODEL
from rag.document_loader import DocumentLoader
from rag.text_splitter import LegalTextSplitter
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.legal_research_agent import LegalResearchAgent
from models.schemas import LegalResearchAnswer

def test_ollama_agent():
    print("==================================================")
    print("TESTING OLLAMA QWEN2.5:1.5B STRUCTURED OUTPUT PIPELINE")
    print("==================================================")

    # 1. Initialize retriever
    vector_store = VectorStoreManager(VECTORSTORE_DIR)
    embedding_service = EmbeddingService(provider="local")
    retriever = RAGRetriever(vector_store, embedding_service)

    # 2. Instantiate LLM service with Ollama provider
    llm_service = LLMService(provider="ollama", model_name=OLLAMA_MODEL)
    agent = LegalResearchAgent(retriever, llm_service)

    query = "What happened in this case?"
    print(f"\nExecuting Agent Workflow for query: '{query}'")
    state = agent.run_workflow(query, top_k=5)

    print("\n--------------------------------------------------")
    print(f"Detected Intent: {state.intent.value}")
    print(f"Retrieved Chunks: {len(state.retrieved_chunks)}")
    print(f"Validation Error: '{state.validation_error}' (Empty string = Success)")
    print(f"Structured Answer Object Present: {state.structured_answer is not None}")

    if state.structured_answer:
        ans: LegalResearchAnswer = state.structured_answer
        print("\n--- GROUNDED ANSWER ---")
        print(ans.answer)
        print("\n--- KEY FINDINGS ---")
        for i, f in enumerate(ans.key_findings, 1):
            print(f"  {i}. {f}")
        print("\n--- EVIDENCE SUMMARY ---")
        for i, e in enumerate(ans.evidence_summary, 1):
            print(f"  {i}. {e}")
        print("\n--- SOURCES ---")
        for i, s in enumerate(ans.sources, 1):
            print(f"  [{i}] {s.document_name} (Page {s.page_number})")

    assert state.structured_answer is not None
    assert isinstance(state.structured_answer, LegalResearchAnswer)
    assert len(state.structured_answer.key_findings) > 0
    assert len(state.structured_answer.sources) > 0
    print("\n==================================================")
    print("OLLAMA STRUCTURED OUTPUT TEST PASSED 100%! ✓")
    print("==================================================")

if __name__ == "__main__":
    test_ollama_agent()
