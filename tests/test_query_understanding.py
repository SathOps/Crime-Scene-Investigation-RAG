import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_DATA_DIR, VECTORSTORE_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from models.schemas import IntentType, LegalResearchAnswer
from rag.document_loader import DocumentLoader
from rag.text_splitter import LegalTextSplitter
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.legal_research_agent import LegalResearchAgent
from utils.query_normalizer import QueryNormalizer
from utils.intent_classifier import IntentClassifier

@dataclass
class TestCase:
    query: str
    expected_intent: IntentType
    category: str
    expected_doc_keywords: List[str]
    is_ambiguous: bool = False
    is_out_of_domain: bool = False

EVALUATION_DATASET: List[TestCase] = [
    # ── Category 1: Correct English ──────────────────────────────────────────
    TestCase(
        query="What did security guard Ankit Sharma state in his deposition?",
        expected_intent=IntentType.WITNESS_ANALYSIS,
        category="Correct English",
        expected_doc_keywords=["Ankit Sharma", "guard", "Apex Towers"]
    ),
    TestCase(
        query="Provide a chronological timeline of events on 13th November 2024.",
        expected_intent=IntentType.TIMELINE,
        category="Correct English",
        expected_doc_keywords=["13th November", "21:10", "21:25", "21:40"]
    ),
    TestCase(
        query="What physical and forensic evidence connects Ramesh Kumar to the crime?",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Correct English",
        expected_doc_keywords=["shell casing", "Item-A", "wristwatch", "CDR"]
    ),
    TestCase(
        query="What legal provisions and sections are invoked under the FIR?",
        expected_intent=IntentType.LEGAL_SECTION_CHECK,
        category="Correct English",
        expected_doc_keywords=["Section 302", "392", "120B", "IPC"]
    ),
    TestCase(
        query="Provide a comprehensive summary of the entire murder case.",
        expected_intent=IntentType.CASE_SUMMARY,
        category="Correct English",
        expected_doc_keywords=["Vikram Malhotra", "Ramesh Kumar", "Apex"]
    ),

    # ── Category 2: Spelling Mistakes ────────────────────────────────────────
    TestCase(
        query="wht did the guard say n does cctv match it?",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Spelling Mistakes",
        expected_doc_keywords=["Ankit Sharma", "CCTV", "Camera"]
    ),
    TestCase(
        query="any diffrnce in witnes statemnts n camera?",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Spelling Mistakes",
        expected_doc_keywords=["witness", "CCTV", "Camera"]
    ),
    TestCase(
        query="show me evrything that happend in order",
        expected_intent=IntentType.TIMELINE,
        category="Spelling Mistakes",
        expected_doc_keywords=["21:10", "21:25", "21:39"]
    ),
    TestCase(
        query="acused aliby at blue dart cafe",
        expected_intent=IntentType.ACCUSED_INFORMATION,
        category="Spelling Mistakes",
        expected_doc_keywords=["Blue Dart Cafe", "Aundh", "Ramesh"]
    ),
    TestCase(
        query="forensik report on 9mm spent shell casng",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Spelling Mistakes",
        expected_doc_keywords=["9mm", "Item-A", "Forensic"]
    ),
    TestCase(
        query="statemnt of wittness priya nair",
        expected_intent=IntentType.WITNESS_ANALYSIS,
        category="Spelling Mistakes",
        expected_doc_keywords=["Priya Nair", "Accountant"]
    ),

    # ── Category 3: Grammatical Errors ───────────────────────────────────────
    TestCase(
        query="when did he left the place?",
        expected_intent=IntentType.TIMELINE,
        category="Grammatical Errors",
        expected_doc_keywords=["21:39", "21:40", "exit"]
    ),
    TestCase(
        query="did they saw the gun at the scene?",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Grammatical Errors",
        expected_doc_keywords=["9mm", "weapon", "casing"]
    ),
    TestCase(
        query="where was he went during the night of murder?",
        expected_intent=IntentType.TIMELINE,
        category="Grammatical Errors",
        expected_doc_keywords=["Shivajinagar", "Blue Dart", "Kothrud"]
    ),
    TestCase(
        query="what happen at 21:25 in building?",
        expected_intent=IntentType.TIMELINE,
        category="Grammatical Errors",
        expected_doc_keywords=["21:25", "bang", "thud"]
    ),
    TestCase(
        query="why does witness statements disagrees?",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Grammatical Errors",
        expected_doc_keywords=["Ankit Sharma", "Priya Nair", "jacket"]
    ),

    # ── Category 4: Informal English & Abbreviations ─────────────────────────
    TestCase(
        query="did their stories match or not?",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Informal English",
        expected_doc_keywords=["witness", "Ankit Sharma", "Priya Nair"]
    ),
    TestCase(
        query="do their stories match?",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Informal English",
        expected_doc_keywords=["witness", "Ankit Sharma", "Priya Nair"]
    ),
    TestCase(
        query="put everything in order by time",
        expected_intent=IntentType.TIMELINE,
        category="Informal English",
        expected_doc_keywords=["21:10", "21:25", "21:40"]
    ),
    TestCase(
        query="sum up the whole case",
        expected_intent=IntentType.CASE_SUMMARY,
        category="Informal English",
        expected_doc_keywords=["Vikram Malhotra", "Ramesh Kumar"]
    ),
    TestCase(
        query="what does the witness say?",
        expected_intent=IntentType.WITNESS_ANALYSIS,
        category="Informal English",
        expected_doc_keywords=["witness", "statement", "Ankit"]
    ),
    TestCase(
        query="give me the tldr of what happened",
        expected_intent=IntentType.CASE_SUMMARY,
        category="Informal English",
        expected_doc_keywords=["Vikram Malhotra", "FIR"]
    ),
    TestCase(
        query="cctv vs guard statements",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Informal English",
        expected_doc_keywords=["Camera", "CCTV", "guard"]
    ),

    # ── Category 5: Paraphrased Questions ────────────────────────────────────
    TestCase(
        query="Can you reconstruct the sequence of incidents hour by hour?",
        expected_intent=IntentType.TIMELINE,
        category="Paraphrased Questions",
        expected_doc_keywords=["20:30", "21:10", "21:25", "21:40"]
    ),
    TestCase(
        query="Highlight any clashes or inconsistencies between eyewitness depositions",
        expected_intent=IntentType.CONTRADICTION_DETECTION,
        category="Paraphrased Questions",
        expected_doc_keywords=["navy blue", "red leather jacket", "Ankit", "Priya"]
    ),
    TestCase(
        query="Under which penal statutes has the accused been charged?",
        expected_intent=IntentType.LEGAL_SECTION_CHECK,
        category="Paraphrased Questions",
        expected_doc_keywords=["Section 302", "392", "IPC"]
    ),
    TestCase(
        query="What did the postmortem examination reveal about the exact cause of death?",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Paraphrased Questions",
        expected_doc_keywords=["gunshot", "postmortem", "hemorrhage"]
    ),

    # ── Category 6: Ambiguous Questions ──────────────────────────────────────
    TestCase(
        query="what about that?",
        expected_intent=IntentType.UNKNOWN,
        category="Ambiguous Questions",
        expected_doc_keywords=[],
        is_ambiguous=True
    ),
    TestCase(
        query="did it?",
        expected_intent=IntentType.UNKNOWN,
        category="Ambiguous Questions",
        expected_doc_keywords=[],
        is_ambiguous=True
    ),
    TestCase(
        query="tell me about him",
        expected_intent=IntentType.UNKNOWN,
        category="Ambiguous Questions",
        expected_doc_keywords=[],
        is_ambiguous=True
    ),
    TestCase(
        query="status?",
        expected_intent=IntentType.UNKNOWN,
        category="Ambiguous Questions",
        expected_doc_keywords=[],
        is_ambiguous=True
    ),

    # ── Category 7: Exact Names, Dates, Legal References ─────────────────────
    TestCase(
        query="What did Call Detail Records show regarding cell tower logs at 21:04 HRS?",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Exact Entities & Dates",
        expected_doc_keywords=["CDR", "Shivajinagar", "21:04"]
    ),
    TestCase(
        query="What charges are framed under Section 302 IPC?",
        expected_intent=IntentType.LEGAL_SECTION_CHECK,
        category="Exact Entities & Dates",
        expected_doc_keywords=["Section 302", "Murder", "IPC"]
    ),
    TestCase(
        query="At what time did Camera 03 go offline on 13th November?",
        expected_intent=IntentType.TIMELINE,
        category="Exact Entities & Dates",
        expected_doc_keywords=["Camera 03", "21:12", "offline"]
    ),
    TestCase(
        query="What was recovered from the accused under Evidence Item-C?",
        expected_intent=IntentType.EVIDENCE_EVALUATION,
        category="Exact Entities & Dates",
        expected_doc_keywords=["Item-C", "wristwatch", "blood"]
    ),

    # ── Category 8: Out-of-Domain / Insufficient Evidence ────────────────────
    TestCase(
        query="Who won the 2024 FIFA World Cup in Qatar?",
        expected_intent=IntentType.GENERAL_RESEARCH,
        category="Out-of-Domain",
        expected_doc_keywords=[],
        is_out_of_domain=True
    ),
    TestCase(
        query="What is the capital city of France?",
        expected_intent=IntentType.GENERAL_RESEARCH,
        category="Out-of-Domain",
        expected_doc_keywords=[],
        is_out_of_domain=True
    ),
    TestCase(
        query="Did the victim own commercial real estate property in London?",
        expected_intent=IntentType.GENERAL_RESEARCH,
        category="Out-of-Domain",
        expected_doc_keywords=[],
        is_out_of_domain=True
    ),
    TestCase(
        query="Was a formal confession recorded before a Magistrate under Section 164 CrPC?",
        expected_intent=IntentType.LEGAL_SECTION_CHECK,
        category="Out-of-Domain",
        expected_doc_keywords=[],
        is_out_of_domain=True
    )
]

def run_evaluation_suite():
    print("=" * 80)
    print("CRIME SCENE INVESTIGATION RAG — QUERY UNDERSTANDING EVALUATION SUITE")
    print(f"Total Test Cases: {len(EVALUATION_DATASET)}")
    print("=" * 80)

    # 1. Setup Vector Store & Knowledge Base
    vector_store = VectorStoreManager(VECTORSTORE_DIR)
    vector_store.clear_store()
    embedding_service = EmbeddingService(provider="local")
    splitter = LegalTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

    sample_files = list(SAMPLE_DATA_DIR.glob("*.txt"))
    for sf in sample_files:
        pages = DocumentLoader.load_file(sf, sf.name)
        chunks = splitter.split_pages(pages)
        vector_store.add_chunks(chunks, embedding_service)

    retriever = RAGRetriever(vector_store, embedding_service)
    llm_service = LLMService(provider="local")
    agent = LegalResearchAgent(retriever, llm_service)

    print(f"Knowledge Base Loaded: {vector_store.get_stats()['total_chunks']} chunks from {len(sample_files)} documents.\n")

    # Metrics Trackers
    baseline_intent_correct = 0
    improved_intent_correct = 0
    improved_retrieval_success = 0
    improved_schema_valid = 0
    improved_grounding_correct = 0
    ambiguity_correctly_handled = 0
    total_latency = 0.0

    category_results: Dict[str, Dict[str, int]] = {}

    print(f"{'#':<3} | {'Category':<22} | {'Original Query':<35} | {'Base Intent':<16} | {'New Intent':<16} | {'Result'}")
    print("-" * 115)

    for idx, tc in enumerate(EVALUATION_DATASET, 1):
        cat = tc.category
        if cat not in category_results:
            category_results[cat] = {"total": 0, "base_intent": 0, "new_intent": 0, "retrieval_ok": 0}
        category_results[cat]["total"] += 1

        # ── 1. BASELINE INTENT CLASSIFICATION (Original Keyword Matching) ──
        q_lower = tc.query.lower()
        if any(w in q_lower for w in ["timeline", "chronology", "sequence of events", "dates", "when did"]):
            base_intent = IntentType.TIMELINE
        elif any(w in q_lower for w in ["contradict", "inconsistent", "discrepancy", "differ", "conflict", "witness statements"]):
            base_intent = IntentType.CONTRADICTION_ANALYSIS
        elif any(w in q_lower for w in ["evidence", "connect", "forensic", "weapon", "blood", "fingerprint"]):
            base_intent = IntentType.EVIDENCE_ANALYSIS
        elif any(w in q_lower for w in ["accused", "suspect", "defendant"]):
            base_intent = IntentType.ACCUSED_INFORMATION
        elif any(w in q_lower for w in ["witness", "statement", "testimony"]):
            base_intent = IntentType.WITNESS_ANALYSIS
        elif any(w in q_lower for w in ["section", "ipc", "crpc", "statute", "act", "provision", "law"]):
            base_intent = IntentType.LEGAL_PROVISIONS
        elif any(w in q_lower for w in ["summary", "what happened", "overview", "brief"]):
            base_intent = IntentType.CASE_SUMMARY
        else:
            base_intent = IntentType.GENERAL_LEGAL_RESEARCH

        base_match = (base_intent == tc.expected_intent)
        if base_match:
            baseline_intent_correct += 1
            category_results[cat]["base_intent"] += 1

        # ── 2. IMPROVED WORKFLOW EXECUTION ──
        start_time = time.time()
        state = agent.run_workflow(tc.query, top_k=3)
        latency = time.time() - start_time
        total_latency += latency

        new_intent_match = (state.intent == tc.expected_intent)
        if new_intent_match:
            improved_intent_correct += 1
            category_results[cat]["new_intent"] += 1

        # Retrieval check
        if tc.is_out_of_domain:
            retrieval_ok = (state.evidence_strength == "Insufficient" or len(state.retrieved_chunks) == 0)
        elif tc.is_ambiguous:
            retrieval_ok = state.is_ambiguous
        else:
            retrieval_ok = any(
                any(kw.lower() in c.text.lower() or kw.lower() in c.document_name.lower() for kw in tc.expected_doc_keywords)
                for c in state.retrieved_chunks
            ) if state.retrieved_chunks else False

        if retrieval_ok:
            improved_retrieval_success += 1
            category_results[cat]["retrieval_ok"] += 1

        # Schema & Grounding check
        if isinstance(state.structured_answer, LegalResearchAnswer):
            improved_schema_valid += 1
            
            # Grounding check for out-of-domain: MUST not hallucinate facts
            if tc.is_out_of_domain:
                grounding_ok = "could not find sufficient supporting information" in state.structured_answer.answer.lower() or state.evidence_strength == "Insufficient"
            elif tc.is_ambiguous:
                grounding_ok = state.is_ambiguous and bool(state.clarification_prompt)
            else:
                grounding_ok = len(state.structured_answer.key_findings) > 0 and len(state.sources) > 0
            
            if grounding_ok:
                improved_grounding_correct += 1

        status_flag = "✓ PASS" if (new_intent_match and retrieval_ok) else "✗ FAIL"
        disp_q = tc.query[:32] + "..." if len(tc.query) > 35 else tc.query
        print(f"{idx:<3} | {cat:<22} | {disp_q:<35} | {base_intent.value[:14]:<16} | {state.intent.value[:14]:<16} | {status_flag}")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY & BREAKDOWN BY CATEGORY")
    print("=" * 80)
    print(f"{'Category':<26} | {'Total':<6} | {'Base Intent Acc':<16} | {'New Intent Acc':<16} | {'Retrieval Coverage'}")
    print("-" * 88)

    for cat, res in category_results.items():
        base_pct = (res["base_intent"] / res["total"]) * 100
        new_pct = (res["new_intent"] / res["total"]) * 100
        ret_pct = (res["retrieval_ok"] / res["total"]) * 100
        print(f"{cat:<26} | {res['total']:<6} | {base_pct:>6.1f}%          | {new_pct:>6.1f}%          | {ret_pct:>6.1f}%")

    total_n = len(EVALUATION_DATASET)
    base_total_pct = (baseline_intent_correct / total_n) * 100
    new_total_pct = (improved_intent_correct / total_n) * 100
    ret_total_pct = (improved_retrieval_success / total_n) * 100
    ground_total_pct = (improved_grounding_correct / total_n) * 100
    schema_total_pct = (improved_schema_valid / total_n) * 100
    avg_latency = (total_latency / total_n)

    print("-" * 88)
    print(f"{'OVERALL TOTAL':<26} | {total_n:<6} | {base_total_pct:>6.1f}%          | {new_total_pct:>6.1f}%          | {ret_total_pct:>6.1f}%\n")

    print("=" * 80)
    print("COMPARATIVE EVALUATION METRICS:")
    print(f"1. Intent Classification Accuracy: {base_total_pct:.1f}% (Baseline) -> {new_total_pct:.1f}% (Improved)  [+{new_total_pct - base_total_pct:.1f}% GAIN]")
    print(f"2. Retrieval Relevance / Evidence Coverage: {ret_total_pct:.1f}%")
    print(f"3. Evidence Grounding & Anti-Hallucination Rate: {ground_total_pct:.1f}%")
    print(f"4. Pydantic Structured Schema Validity: {schema_total_pct:.1f}%")
    print(f"5. Average Latency per Query: {avg_latency:.3f} seconds (Fast deterministic normalization)")
    print("=" * 80)

    assert new_total_pct >= 90.0, f"Expected intent accuracy >= 90%, got {new_total_pct}%"
    assert ret_total_pct >= 85.0, f"Expected retrieval coverage >= 85%, got {ret_total_pct}%"
    assert schema_total_pct == 100.0, f"Expected 100% Pydantic schema validity, got {schema_total_pct}%"
    print("\n✓ ALL EVALUATION SUITE ASSERTIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_evaluation_suite()
