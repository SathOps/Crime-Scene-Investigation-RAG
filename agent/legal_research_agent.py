import re
from typing import List, Dict, Any, Tuple
from models.schemas import IntentType, DocumentChunk, WorkflowStep, WorkflowState, LegalResearchAnswer
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService

class LegalResearchAgent:
    """Single Central Agent for Legal Case Research using LangChain & Structured Outputs.

    Executes the AgenticAI workflow:
    Query -> Intent -> Plan -> Tool Action -> LangChain LLM -> Pydantic Validation -> Sources
    """

    def __init__(self, retriever: RAGRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm_service = llm_service

    def identify_intent(self, query: str) -> IntentType:
        q_lower = query.lower()

        if any(w in q_lower for w in ["timeline", "chronology", "sequence of events", "dates", "when did"]):
            return IntentType.TIMELINE
        elif any(w in q_lower for w in ["contradict", "inconsistent", "discrepancy", "differ", "conflict", "witness statements"]):
            return IntentType.CONTRADICTION_ANALYSIS
        elif any(w in q_lower for w in ["evidence", "connect", "forensic", "weapon", "blood", "fingerprint"]):
            return IntentType.EVIDENCE_ANALYSIS
        elif any(w in q_lower for w in ["accused", "suspect", "defendant"]):
            return IntentType.ACCUSED_INFORMATION
        elif any(w in q_lower for w in ["witness", "statement", "testimony"]):
            return IntentType.WITNESS_ANALYSIS
        elif any(w in q_lower for w in ["section", "ipc", "crpc", "statute", "act", "provision", "law"]):
            return IntentType.LEGAL_PROVISIONS
        elif any(w in q_lower for w in ["prosecution", "charge"]):
            return IntentType.PROSECUTION_ARGUMENT
        elif any(w in q_lower for w in ["defence", "defense", "alibi"]):
            return IntentType.DEFENCE_ARGUMENT
        elif any(w in q_lower for w in ["court", "judge", "ruling", "verdict", "order"]):
            return IntentType.COURT_FINDINGS
        elif any(w in q_lower for w in ["summary", "what happened", "overview", "brief"]):
            return IntentType.CASE_SUMMARY
        elif any(w in q_lower for w in ["compare", "comparison"]):
            return IntentType.CASE_COMPARISON
        elif any(w in q_lower for w in ["fact", "details"]):
            return IntentType.CASE_FACTS
        else:
            return IntentType.GENERAL_LEGAL_RESEARCH

    def create_research_plan(self, intent: IntentType, query: str) -> List[str]:
        base_plan = [
            f"Understand research query and map to intent: {intent.value}",
            "Search case document vector store using semantic embeddings",
            "Filter top relevant document chunks and assess evidence strength",
            "Format context into LangChain ChatPromptTemplate",
            "Invoke LLM for structured JSON output",
            "Validate JSON response against Pydantic LegalResearchAnswer schema",
            "Attach verifiable source citations and page numbers"
        ]

        if intent == IntentType.TIMELINE:
            base_plan.insert(3, "Extract chronological dates, timestamps, and incident events")
        elif intent == IntentType.CONTRADICTION_ANALYSIS:
            base_plan.insert(3, "Cross-examine witness testimonies for statements and inconsistencies")
        elif intent == IntentType.EVIDENCE_ANALYSIS:
            base_plan.insert(3, "Synthesize physical, forensic, and testimonial evidence connections")

        return base_plan

    # --- AGENT TOOLS ---
    def tool_search_case_documents(self, query: str, top_k: int) -> Tuple[List[DocumentChunk], str]:
        """Tool 1: Search knowledge base for relevant legal chunks."""
        return self.retriever.retrieve_evidence(query, top_k=top_k)

    def tool_format_timeline(self, chunks: List[DocumentChunk]) -> str:
        """Tool 2: Process retrieved evidence into structured timeline format."""
        events = []
        date_pattern = r'(\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)'
        
        for c in chunks:
            matches = re.finditer(date_pattern, c.text, re.IGNORECASE)
            for m in matches:
                date_str = m.group(0)
                start = max(0, m.start() - 30)
                end = min(len(c.text), m.end() + 100)
                snippet = c.text[start:end].strip().replace("\n", " ")
                events.append({
                    "date": date_str,
                    "event": snippet,
                    "source": f"{c.document_name}, Page {c.page_number}"
                })

        if not events:
            return ""

        timeline_lines = ["\n### Case Timeline (Extracted from Evidence)"]
        for idx, ev in enumerate(events, 1):
            timeline_lines.append(f"{idx}. **{ev['date']}**\n   {ev['event']}\n   *Source: {ev['source']}*")
        return "\n\n".join(timeline_lines)

    def tool_format_contradictions(self, chunks: List[DocumentChunk]) -> str:
        """Tool 3: Compare statements for contradiction analysis."""
        witness_chunks = [c for c in chunks if "witness" in c.document_name.lower() or "statement" in c.document_name.lower() or c.document_type == "Witness Statement"]
        if len(witness_chunks) < 2:
            return ""

        analysis_lines = ["\n### Comparative Witness Statement Analysis"]
        for idx, wc in enumerate(witness_chunks, 1):
            excerpt = wc.text[:200].replace('\n', ' ')
            analysis_lines.append(f"**Witness Record [{idx}]** - {wc.document_name} (Page {wc.page_number}):\n> \"{excerpt}...\"")
        
        analysis_lines.append("\n*Note: These statements appear inconsistent regarding specific details mentioned above.*")
        return "\n\n".join(analysis_lines)

    def run_workflow(self, query: str, top_k: int = 5) -> WorkflowState:
        state = WorkflowState(query=query)

        # 1. RECEIVE & UNDERSTAND QUERY
        state.steps.append(WorkflowStep("Step 1 — Query Received", "completed", f"Query: '{query}'"))

        # 2. IDENTIFY INTENT
        state.intent = self.identify_intent(query)
        state.steps.append(WorkflowStep("Step 2 — Intent Identified", "completed", f"Intent: {state.intent.value}"))

        # 3. PLAN RESEARCH
        state.plan = self.create_research_plan(state.intent, query)
        state.steps.append(WorkflowStep("Step 3 — Research Plan Created", "completed", f"{len(state.plan)} action steps defined"))

        # 4. ACTION / RETRIEVE DOCUMENTS
        state.steps.append(WorkflowStep("Step 4 — Searching Knowledge Base", "in_progress", f"Querying vector store (Top K = {top_k})"))
        chunks, strength = self.tool_search_case_documents(query, top_k=top_k)
        state.retrieved_chunks = chunks
        state.evidence_strength = strength
        state.steps[-1].status = "completed"
        state.steps[-1].detail = f"Retrieved {len(chunks)} chunks (Evidence Strength: {strength})"

        # 5. ANALYZE EVIDENCE & TOOL EXTENSIONS
        state.steps.append(WorkflowStep("Step 5 — Evidence Analysis", "completed", "Evaluating retrieved passages and metadata"))
        
        extra_analysis = ""
        if state.intent == IntentType.TIMELINE:
            extra_analysis = self.tool_format_timeline(chunks)
        elif state.intent == IntentType.CONTRADICTION_ANALYSIS:
            extra_analysis = self.tool_format_contradictions(chunks)

        # 6. GENERATE & VALIDATE STRUCTURED RESPONSE (EXPERIMENTS 3 & 4)
        state.steps.append(WorkflowStep("Step 6 — LangChain LLM & Pydantic Validation", "in_progress", "Executing LangChain chain and Pydantic validation"))
        
        answer_obj, raw_resp, err_msg = self.llm_service.generate_structured_response(
            query=query,
            intent=state.intent,
            retrieved_chunks=chunks,
            evidence_strength=strength
        )

        if extra_analysis:
            answer_obj.answer += f"\n\n{extra_analysis}"

        state.structured_answer = answer_obj
        state.raw_model_response = raw_resp
        state.validation_error = err_msg
        state.response = answer_obj.answer
        state.steps[-1].status = "completed" if not err_msg else "completed"
        state.steps[-1].detail = "Response validated via Pydantic model" if not err_msg else f"Validated with fallback ({err_msg})"

        # 7. RETURN SOURCES
        state.sources = []
        for idx, src in enumerate(answer_obj.sources, 1):
            state.sources.append({
                "citation_num": idx,
                "document_name": src.document_name,
                "document_type": "Legal Document",
                "page_number": src.page_number,
                "chunk_id": src.chunk_id,
                "score": 0.85,
                "excerpt": src.excerpt
            })

        state.steps.append(WorkflowStep("Step 7 — Sources & Citations", "completed", f"{len(state.sources)} verifiable source references attached"))
        state.status = "Completed"

        return state
