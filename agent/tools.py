"""Stateless, session-safe LangChain tools for legal case research."""
from typing import List
from langchain_core.tools import tool
from utils.query_normalizer import QueryNormalizer
from models.schemas import DocumentChunk
from agent.timeline_pipeline import TimelinePipeline
from agent.contradiction_pipeline import ContradictionPipeline


def text_to_document_chunks(text: str) -> List[DocumentChunk]:
    """Parse formatted context string or raw text into DocumentChunk objects."""
    import re
    if not text or not text.strip():
        return []

    items = re.findall(
        r'(?:---\s*EVIDENCE ITEM \d+:?\s*([^\n]+?)(?:\s*---)?\n(.*?)(?=(?:---\s*EVIDENCE ITEM|\Z)))',
        text,
        re.DOTALL
    )
    if items:
        chunks = []
        for idx, (header, body) in enumerate(items, 1):
            doc_name = "Retrieved Case Record"
            page_num = 1
            chunk_id = f"item_{idx:02d}"

            doc_match = re.search(r'([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx))', header)
            if doc_match:
                doc_name = doc_match.group(1)
            page_match = re.search(r'(?:Page|p\.)\s*(\d+)', header)
            if page_match:
                page_num = int(page_match.group(1))
            chunk_match = re.search(r'Chunk\s*#?([a-zA-Z0-9_\-]+)', header)
            if chunk_match:
                chunk_id = chunk_match.group(1)

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_name=doc_name,
                document_type=doc_name.split("_")[-1].replace(".txt", "") if "_" in doc_name else "Case Record",
                page_number=page_num,
                text=body.strip()
            ))
        return chunks

    return [DocumentChunk(
        chunk_id="chunk_01",
        document_name="Retrieved Case Record",
        document_type="Case Record",
        page_number=1,
        text=text.strip()
    )]


def build_legal_tools(retriever):
    """Build tools bound to one retriever instance; evidence is passed per invocation."""

    @tool
    def retrieve_case_evidence(query: str, top_k: int = 5) -> str:
        """Retrieve evidence from this research session's case-file vector store.

        Args:
            query: Search terms describing the evidence needed.
            top_k: Maximum number of passages to return.
        """
        norm = QueryNormalizer.normalize(query)
        chunks, strength = retriever.retrieve_evidence(
            query=query,
            top_k=max(1, min(int(top_k), 12)),
            normalized_query=norm.normalized_query,
            entities=norm.entities_preserved,
        )
        if not chunks:
            return "No relevant evidence chunks were found in the current case-file index."
        return "Evidence relevance (not proof strength): " + strength + "\n\n" + retriever.format_chunks_for_context(chunks)

    @tool
    def extract_timeline_events(evidence_text: str) -> str:
        """Extract dates and times from supplied retrieved evidence and order them chronologically.

        Args:
            evidence_text: Retrieved source passages, including document and page labels.
        """
        import re
        text = (evidence_text or "").strip()
        if not text:
            return "No retrieved evidence was supplied; no timeline can be extracted."

        parsed_chunks = text_to_document_chunks(text)
        result = TimelinePipeline.extract_events_structured(parsed_chunks)

        # If supplied text lacks sufficient temporal events, and retriever is available,
        # supplement with comprehensive timeline chunks from the session vector store.
        if (result.validation_status == "insufficient" or len(result.events) < 3) and retriever is not None:
            try:
                timeline_chunks, _ = retriever.retrieve_evidence(
                    query="chronological timeline sequence of events dates timestamps CCTV postmortem autopsy entry exit",
                    top_k=8,
                    intent="TIMELINE"
                )
                if timeline_chunks:
                    combined = list(parsed_chunks) + [c for c in timeline_chunks if c.chunk_id not in {p.chunk_id for p in parsed_chunks}]
                    result = TimelinePipeline.extract_events_structured(combined)
            except Exception:
                pass

        if result.validation_status == "insufficient" or not result.events:
            return (
                "Controlled Error: Unable to extract a reliable chronological timeline from the provided evidence. "
                "Insufficient temporal markers or timeline passages found in the supplied records."
            )

        return TimelinePipeline.format_timeline_markdown_table(result)

    @tool
    def analyze_witness_contradictions(evidence_text: str) -> str:
        """Identify and evaluate factual inconsistencies between witness accounts and physical evidence.

        Args:
            evidence_text: Retrieved source passages, including document and page labels.
        """
        text = (evidence_text or "").strip()
        if not text:
            return "No retrieved evidence was supplied; contradiction analysis was not performed."

        parsed_chunks = text_to_document_chunks(text)
        result = ContradictionPipeline.analyze_contradictions(parsed_chunks)

        # If supplied text lacks witness context and retriever is available,
        # supplement with witness statement chunks from the session vector store.
        if result.status == "insufficient_evidence" and retriever is not None:
            try:
                witness_chunks, _ = retriever.retrieve_evidence(
                    query="witness statements depositions contradiction discrepancy CCTV camera clothing entrance",
                    top_k=6,
                    intent="CONTRADICTION_DETECTION"
                )
                if witness_chunks:
                    combined = list(parsed_chunks) + [c for c in witness_chunks if c.chunk_id not in {p.chunk_id for p in parsed_chunks}]
                    result = ContradictionPipeline.analyze_contradictions(combined)
            except Exception:
                pass

        return ContradictionPipeline.format_contradiction_markdown(result)

    @tool
    def summarize_case_facts(evidence_text: str) -> str:
        """Return retrieved excerpts for source-attributed case-fact synthesis.

        Args:
            evidence_text: Retrieved source passages, including document and page labels.
        """
        text = (evidence_text or "").strip()
        if not text:
            return "No retrieved case evidence is available to summarize."
        return (
            "Retrieved source excerpts follow. Preserve each source's procedural role and "
            "do not treat allegations, testimony, forensic findings, and judicial findings as interchangeable. "
            "These excerpts do not independently establish guilt.\n\n" + text[:16000]
        )

    return [
        retrieve_case_evidence,
        extract_timeline_events,
        analyze_witness_contradictions,
        summarize_case_facts,
    ]


# Backwards-compatible registry for integrations that inspect available tool names.
# Runtime agents should use build_legal_tools(retriever) to bind their own retriever.
LEGAL_TOOLS = []
TOOLS_BY_NAME = {}
