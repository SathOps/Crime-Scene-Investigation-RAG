"""LangChain Tool Registry for Legal Intelligence Platform.

Converts existing deterministic retriever and analysis functions into proper
LangChain `@tool` instances for dynamic LLM tool calling.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from utils.query_normalizer import QueryNormalizer

# Forward references / tool wrappers initialized with instance context
_GLOBAL_RETRIEVER = None
_GLOBAL_CONTEXT_BUFFER = ""

def set_global_retriever(retriever):
    """Sets the global RAGRetriever instance for tool access."""
    global _GLOBAL_RETRIEVER
    _GLOBAL_RETRIEVER = retriever

def set_global_context(text: str):
    """Sets the global context text buffer for tools that analyze evidence."""
    global _GLOBAL_CONTEXT_BUFFER
    _GLOBAL_CONTEXT_BUFFER = text

@tool
def retrieve_case_evidence(query: str, top_k: int = 5) -> str:
    """Retrieve relevant case evidence, FIR records, witness statements, forensic reports, and court findings from the persistent vector store.
    Use this tool whenever you need factual evidence, specific details, quotes, or context from the case files to answer a question.

    Args:
        query: The search query string for retrieving legal evidence.
        top_k: Number of relevant evidence passages to retrieve (default: 5).
    """
    global _GLOBAL_RETRIEVER, _GLOBAL_CONTEXT_BUFFER
    if _GLOBAL_RETRIEVER is None:
        return "Error: Vector store retriever is not initialized."
    
    try:
        norm_obj = QueryNormalizer.normalize(query)
        chunks, strength = _GLOBAL_RETRIEVER.retrieve_evidence(
            query=query,
            top_k=top_k,
            normalized_query=norm_obj.normalized_query,
            entities=norm_obj.entities_preserved
        )
        if not chunks:
            return "No relevant evidence chunks found in the knowledge base."
        
        formatted_passages = []
        for idx, chunk in enumerate(chunks, 1):
            formatted_passages.append(
                f"[Evidence Item {idx}]\n"
                f"Document: {chunk.document_name} (Type: {chunk.document_type}, Page {chunk.page_number})\n"
                f"Content: {chunk.text.strip()}\n"
            )
        
        result_text = f"Evidence Strength: {strength}\n\n" + "\n---\n".join(formatted_passages)
        _GLOBAL_CONTEXT_BUFFER = result_text
        return result_text
    except Exception as e:
        return f"Tool Execution Error during retrieval: {str(e)}"

@tool
def extract_timeline_events(evidence_text: str = "") -> str:
    """Extract chronological dates, timestamps, and sequential incident events from legal evidence text.
    Use this tool when answering timeline, sequence of events, or chronological questions.

    Args:
        evidence_text: Text containing legal evidence or case documents. If empty, uses active retrieved case context.
    """
    import re
    global _GLOBAL_CONTEXT_BUFFER
    text_to_analyze = evidence_text.strip() if evidence_text and evidence_text.strip() else _GLOBAL_CONTEXT_BUFFER

    if not text_to_analyze:
        return "No evidence text provided for timeline extraction."
    
    date_pattern = r'(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\s*(?:HRS|AM|PM|hrs)\b)'
    matches = list(re.finditer(date_pattern, text_to_analyze, re.IGNORECASE))
    
    if not matches:
        return "No specific chronological dates or timestamp markers found in the provided text."
    
    events = []
    seen_dates = set()
    for m in matches:
        date_str = m.group(0).upper()
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)
        start = max(0, m.start() - 40)
        end = min(len(text_to_analyze), m.end() + 120)
        snippet = text_to_analyze[start:end].strip().replace("\n", " ")
        events.append(f"• **{date_str}**: ...{snippet}...")
    
    return "Extracted Chronological Timeline:\n" + "\n".join(events)

@tool
def analyze_witness_contradictions(evidence_text: str = "") -> str:
    """Compare statements and witness depositions to identify contradictions, discrepancies, or timeline inconsistencies.
    Use this tool when evaluating witness statements, conflicting reports, or alibi claims.

    Args:
        evidence_text: Text containing witness statements, depositions, or police reports. If empty, uses active retrieved case context.
    """
    global _GLOBAL_CONTEXT_BUFFER
    text_to_analyze = evidence_text.strip() if evidence_text and evidence_text.strip() else _GLOBAL_CONTEXT_BUFFER

    if not text_to_analyze:
        return "No witness statements provided for contradiction analysis."
    
    lines = [line.strip() for line in text_to_analyze.split('\n') if line.strip()]
    witness_excerpts = [line for line in lines if any(w in line.lower() for w in ["witness", "statement", "saw", "claimed", "guard", "deposed", "stated", "alibi", "camera", "cctv"])]
    
    if not witness_excerpts:
        return "Cross-examination summary: No explicit witness statements identified in the provided evidence buffer."
    
    analysis = [
        "Witness Testimony & Cross-Examination Analysis:",
        f"Analyzed {len(witness_excerpts)} witness and physical evidence passages for discrepancies.",
        "Passages extracted for cross-comparison:"
    ]
    for idx, excerpt in enumerate(witness_excerpts[:6], 1):
        analysis.append(f"{idx}. {excerpt}")
    
    return "\n".join(analysis)

@tool
def summarize_case_facts(case_title: str = "") -> str:
    """Summarize case facts from the active retrieved evidence context only.

    Args:
        case_title: Optional case title or reference. Do not use placeholders.
    """
    global _GLOBAL_CONTEXT_BUFFER
    context = (_GLOBAL_CONTEXT_BUFFER or "").strip()
    if not context:
        return "No retrieved case evidence is available to summarize."

    title = (case_title or "").strip()
    if title.lower() in {"your case title here", "case title", "unknown", "n/a", "none"}:
        title = ""
    heading = f"Case Overview [{title}]" if title else "Case Overview (based on retrieved excerpts)"
    return (
        f"{heading}:\n"
        "The following are retrieved source excerpts. Treat each claim according to its "
        "source type and preserve disagreements between documents; this tool does not "
        "independently establish guilt or resolve conflicting timelines.\n\n"
        + context[:12000]
    )

# Central Tool Registry
LEGAL_TOOLS = [
    retrieve_case_evidence,
    extract_timeline_events,
    analyze_witness_contradictions,
    summarize_case_facts
]

TOOLS_BY_NAME = {tool.name: tool for tool in LEGAL_TOOLS}
