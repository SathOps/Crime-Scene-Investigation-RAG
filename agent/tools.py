"""Stateless, session-safe LangChain tools for legal case research."""
from typing import List
from langchain_core.tools import tool
from utils.query_normalizer import QueryNormalizer


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
        """Extract dates and times from supplied retrieved evidence.

        Args:
            evidence_text: Retrieved source passages, including document and page labels.
        """
        import re
        text = (evidence_text or "").strip()
        if not text:
            return "No retrieved evidence was supplied; no timeline can be extracted."
        pattern = r'(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\s*(?:HRS|AM|PM)\b)'
        events, seen = [], set()
        for match in re.finditer(pattern, text, re.IGNORECASE):
            stamp = match.group(0)
            if stamp.lower() in seen:
                continue
            seen.add(stamp.lower())
            start, end = max(0, match.start()-100), min(len(text), match.end()+160)
            events.append(f"- {stamp}: {text[start:end].replace(chr(10), ' ').strip()}")
        return "\n".join(events) if events else "No explicit dates or timestamps were found in the supplied passages."

    @tool
    def analyze_witness_contradictions(evidence_text: str) -> str:
        """Identify potentially inconsistent witness and physical-evidence passages.

        Args:
            evidence_text: Retrieved source passages, including document and page labels.
        """
        text = (evidence_text or "").strip()
        if not text:
            return "No retrieved evidence was supplied; contradiction analysis was not performed."
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        relevant = [line for line in lines if any(term in line.lower() for term in (
            "witness", "statement", "deposed", "testified", "cctv", "camera", "alibi", "observed", "recorded"
        ))]
        if not relevant:
            return "No clearly labelled witness or physical-evidence passages were identified."
        return (
            "Potentially relevant passages for comparison (not a finding that they contradict):\n"
            + "\n".join(f"- {line}" for line in relevant[:12])
            + "\nA contradiction must be established by comparing the underlying claims, sources, and time/context."
        )

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
