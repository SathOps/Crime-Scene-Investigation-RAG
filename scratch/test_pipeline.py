import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_DATA_DIR, VECTORSTORE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from rag.document_loader import DocumentLoader
from rag.text_splitter import LegalTextSplitter
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.legal_research_agent import LegalResearchAgent
from models.schemas import IntentType, LegalResearchAnswer

def run_tests():
    print("==================================================")
    print("STARTING EXPERIMENT 3 + EXPERIMENT 4 VERIFICATION TESTS")
    print("==================================================")

    # 1. Initialize Vector Store & Services
    vector_store = VectorStoreManager(VECTORSTORE_DIR)
    vector_store.clear_store()
    print("✓ Vector store initialized and cleared.")

    embedding_service = EmbeddingService(provider="local")
    splitter = LegalTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

    # 2. Ingest Sample Files
    sample_files = list(SAMPLE_DATA_DIR.glob("*.txt"))
    total_chunks = 0
    for sf in sample_files:
        pages = DocumentLoader.load_file(sf, sf.name)
        chunks = splitter.split_pages(pages)
        vector_store.add_chunks(chunks, embedding_service)
        total_chunks += len(chunks)
        print(f"  - Ingested '{sf.name}': {len(pages)} page(s), {len(chunks)} chunk(s)")

    stats = vector_store.get_stats()
    print(f"✓ Knowledge Base Ready: {stats['total_documents']} docs, {stats['total_chunks']} total chunks.")
    assert stats['total_chunks'] > 0

    # 3. Instantiate Agent with Local Provider (for instant test verification without external API dependence)
    retriever = RAGRetriever(vector_store, embedding_service)
    llm_service = LLMService(provider="local")
    agent = LegalResearchAgent(retriever, llm_service)

    # TEST 1: Summary Query & Pydantic Validation
    print("\n--------------------------------------------------")
    print("TEST 1: Case Summary Query & Pydantic Validation")
    q1 = "What happened in this case?"
    state1 = agent.run_workflow(q1, top_k=3)
    print(f"Query: '{q1}'")
    print(f"Detected Intent: {state1.intent.value}")
    print(f"Pydantic Validation: {'Passed' if state1.structured_answer else 'Failed'}")
    assert state1.intent == IntentType.CASE_SUMMARY
    assert isinstance(state1.structured_answer, LegalResearchAnswer)
    assert len(state1.structured_answer.key_findings) > 0

    # TEST 2: Evidence Analysis & Pydantic Output
    print("\n--------------------------------------------------")
    print("TEST 2: Evidence Analysis & Pydantic Model Output")
    q2 = "What evidence connects the accused to the crime?"
    state2 = agent.run_workflow(q2, top_k=3)
    print(f"Query: '{q2}'")
    print(f"Key Findings Count: {len(state2.structured_answer.key_findings)}")
    print(f"Sources Count: {len(state2.structured_answer.sources)}")
    assert state2.intent == IntentType.EVIDENCE_ANALYSIS
    assert len(state2.structured_answer.sources) > 0

    # TEST 3: Timeline Extraction Query
    print("\n--------------------------------------------------")
    print("TEST 3: Timeline Extraction Query")
    q3 = "Give me the timeline of events."
    state3 = agent.run_workflow(q3, top_k=3)
    print(f"Query: '{q3}'")
    assert state3.intent == IntentType.TIMELINE

    # TEST 4: Contradiction Analysis Query
    print("\n--------------------------------------------------")
    print("TEST 4: Contradiction Analysis Query")
    q4 = "Are there contradictions between the witness statements?"
    state4 = agent.run_workflow(q4, top_k=3)
    print(f"Query: '{q4}'")
    assert state4.intent == IntentType.CONTRADICTION_ANALYSIS

    # TEST 5: Out-of-Domain Anti-Hallucination Safeguard
    print("\n--------------------------------------------------")
    print("TEST 5: Out-of-Domain Anti-Hallucination Safeguard")
    q5 = "Who won the 2024 FIFA World Cup in Qatar?"
    state5 = agent.run_workflow(q5, top_k=3)
    print(f"Query: '{q5}'")
    print(f"Answer: {state5.structured_answer.answer}")
    assert "could not find sufficient supporting information" in state5.structured_answer.answer.lower() or state5.evidence_strength in ["Low", "Insufficient"]

    print("\n==================================================")
    print("ALL EXPERIMENT 3 + 4 TESTS PASSED SUCCESSFULLY! ✓")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
