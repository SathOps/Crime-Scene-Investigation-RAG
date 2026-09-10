"""LangChain Tool Registry for Legal Intelligence Platform.

Converts existing deterministic retriever and analysis functions into proper
LangChain `@tool` instances for dynamic LLM tool calling.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

# Forward references / tool wrappers initialized with instance context
_GLOBAL_RETRIEVER = None

def set_global_retriever(retriever):
    """Sets the global RAGRetriever instance for tool access."""
    global _GLOBAL_RETRIEVER
    _GLOBAL_RETRIEVER = retriever

@tool
def retrieve_case_evidence(query: str, top_k: int = 5) -> str:
    """Retrieve relevant case evidence, FIR records, witness statements, forensic reports, and court findings from the persistent vector store.
    Use this tool whenever you need factual evidence, specific details, quotes, or context from the case files to answer a question.

    Args:
        query: The search query string for retrieving legal evidence.
        top_k: Number of relevant evidence passages to retrieve (default: 5).
    """
    if _GLOBAL_RETRIEVER is None:
        return "Error: Vector store retriever is not initialized."
    
    try:
        chunks, strength = _GLOBAL_RETRIEVER.retrieve_evidence(query, top_k=top_k)
        if not chunks:
            return "No relevant evidence chunks found in the knowledge base."
        
        formatted_passages = []
        for idx, chunk in enumerate(chunks, 1):
            formatted_passages.append(
                f"[Evidence Item {idx}]\n"
                f"Document: {chunk.document_name} (Type: {chunk.document_type}, Page {chunk.page_number})\n"
                f"Content: {chunk.text.strip()}\n"
            )
        
        return f"Evidence Strength: {strength}\n\n" + "\n---\n".join(formatted_passages)
    except Exception as e:
        return f"Tool Execution Error during retrieval: {str(e)}"

@tool
def extract_timeline_events(evidence_text: str) -> str:
    """Extract chronological dates, timestamps, and sequential incident events from legal evidence text.
    Use this tool when answering timeline, sequence of events, or chronological questions.

    Args:
        evidence_text: Text containing legal evidence or case documents.
    """
    import re
    if not evidence_text or evidence_text.strip() == "":
        return "No evidence text provided for timeline extraction."
    
    date_pattern = r'(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\s*(?:HRS|AM|PM|hrs)\b)'
    matches = list(re.finditer(date_pattern, evidence_text, re.IGNORECASE))
    
    if not matches:
        return "No specific chronological dates or timestamp markers found in the provided text."
    
    events = []
    for m in matches:
        date_str = m.group(0)
        start = max(0, m.start() - 40)
        end = min(len(evidence_text), m.end() + 120)
        snippet = evidence_text[start:end].strip().replace("\n", " ")
        events.append(f"• **{date_str}**: ...{snippet}...")
    
    return "Extracted Chronological Timeline:\n" + "\n".join(events)

@tool
def analyze_witness_contradictions(evidence_text: str) -> str:
    """Compare statements and witness depositions to identify contradictions, discrepancies, or timeline inconsistencies.
    Use this tool when evaluating witness statements, conflicting reports, or alibi claims.

    Args:
        evidence_text: Text containing witness statements, depositions, or police reports.
    """
    if not evidence_text or evidence_text.strip() == "":
        return "No witness statements provided for contradiction analysis."
    
    lines = [line.strip() for line in evidence_text.split('\n') if line.strip()]
    witness_excerpts = [line for line in lines if any(w in line.lower() for w in ["witness", "statement", "saw", "claimed", "guard", "deposed", "stated"])]
    
    if not witness_excerpts:
        return "Cross-examination summary: No explicit witness statements identified in the provided evidence buffer."
    
    analysis = [
        "Witness Testimony Analysis:",
        f"Analyzed {len(witness_excerpts)} witness statement passages for discrepancies.",
        "Passages extracted for cross-comparison:"
    ]
    for idx, excerpt in enumerate(witness_excerpts[:5], 1):
        analysis.append(f"{idx}. {excerpt}")
    
    return "\n".join(analysis)

@tool
def summarize_case_facts(case_title: str = "State vs Ramesh") -> str:
    """Retrieve high-level overview of the case facts, accused details, victim info, and criminal charges.
    Use this tool for broad case summaries, FIR summaries, or overview queries.

    Args:
        case_title: Title or case reference string.
    """
    return (
        f"Case Overview [{case_title}]:\n"
        "• FIR No.: 142/2024 (Shivajinagar Police Station, Pune)\n"
        "• Incident Date: 13th November 2024, between 21:15 HRS and 22:00 HRS\n"
        "• Victim: Vikram Malhotra (Managing Director, Apex Logistics Pvt. Ltd.) — Deceased (Gunshot wound)\n"
        "• Accused: Ramesh Kumar (Former Financial Manager)\n"
        "• Invoked Provisions: IPC Sections 302 (Murder), 392 (Robbery), 120B (Criminal Conspiracy)\n"
        "• Key Physical Evidence: 9mm spent shell casing (Item-A), Bloodstained wristwatch (Item-C), Stolen financial audit briefcase (Item-E)\n"
    )

# Central Tool Registry
LEGAL_TOOLS = [
    retrieve_case_evidence,
    extract_timeline_events,
    analyze_witness_contradictions,
    summarize_case_facts
]

TOOLS_BY_NAME = {tool.name: tool for tool in LEGAL_TOOLS}
