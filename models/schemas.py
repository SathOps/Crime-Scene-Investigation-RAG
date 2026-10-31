from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IntentType(str, Enum):
    CASE_SUMMARY = "CASE_SUMMARY"
    CASE_FACTS = "CASE_FACTS"
    ACCUSED_INFORMATION = "ACCUSED_INFORMATION"
    EVIDENCE_ANALYSIS = "EVIDENCE_ANALYSIS"
    EVIDENCE_EVALUATION = "EVIDENCE_EVALUATION"
    WITNESS_ANALYSIS = "WITNESS_ANALYSIS"
    TIMELINE = "TIMELINE"
    LEGAL_PROVISIONS = "LEGAL_PROVISIONS"
    LEGAL_SECTION_CHECK = "LEGAL_SECTION_CHECK"
    PROSECUTION_ARGUMENT = "PROSECUTION_ARGUMENT"
    DEFENCE_ARGUMENT = "DEFENCE_ARGUMENT"
    COURT_FINDINGS = "COURT_FINDINGS"
    CONTRADICTION_ANALYSIS = "CONTRADICTION_ANALYSIS"
    CONTRADICTION_DETECTION = "CONTRADICTION_DETECTION"
    CASE_COMPARISON = "CASE_COMPARISON"
    GENERAL_LEGAL_RESEARCH = "GENERAL_LEGAL_RESEARCH"
    GENERAL_RESEARCH = "GENERAL_RESEARCH"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def normalize_intent(cls, val: Any) -> "IntentType":
        if isinstance(val, cls):
            return val
        s = str(val).strip().upper()
        alias_map = {
            "CONTRADICTION_DETECTION": cls.CONTRADICTION_ANALYSIS,
            "CONTRADICTION_ANALYSIS": cls.CONTRADICTION_ANALYSIS,
            "EVIDENCE_EVALUATION": cls.EVIDENCE_ANALYSIS,
            "EVIDENCE_ANALYSIS": cls.EVIDENCE_ANALYSIS,
            "LEGAL_SECTION_CHECK": cls.LEGAL_PROVISIONS,
            "LEGAL_PROVISIONS": cls.LEGAL_PROVISIONS,
            "GENERAL_RESEARCH": cls.GENERAL_LEGAL_RESEARCH,
            "GENERAL_LEGAL_RESEARCH": cls.GENERAL_LEGAL_RESEARCH,
        }
        if s in alias_map:
            return alias_map[s]
        for item in cls:
            if item.value == s:
                return item
        return cls.UNKNOWN

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        val = getattr(other, "value", other)
        if not isinstance(val, str):
            return False
        if self.value == val:
            return True
        equiv_pairs = {
            "CONTRADICTION_DETECTION": "CONTRADICTION_ANALYSIS",
            "CONTRADICTION_ANALYSIS": "CONTRADICTION_DETECTION",
            "EVIDENCE_EVALUATION": "EVIDENCE_ANALYSIS",
            "EVIDENCE_ANALYSIS": "EVIDENCE_EVALUATION",
            "LEGAL_SECTION_CHECK": "LEGAL_PROVISIONS",
            "LEGAL_PROVISIONS": "LEGAL_SECTION_CHECK",
            "GENERAL_RESEARCH": "GENERAL_LEGAL_RESEARCH",
            "GENERAL_LEGAL_RESEARCH": "GENERAL_RESEARCH"
        }
        return equiv_pairs.get(self.value) == val

    def __hash__(self) -> int:
        canonical_map = {
            "CONTRADICTION_DETECTION": "CONTRADICTION_ANALYSIS",
            "EVIDENCE_EVALUATION": "EVIDENCE_ANALYSIS",
            "LEGAL_SECTION_CHECK": "LEGAL_PROVISIONS",
            "GENERAL_RESEARCH": "GENERAL_LEGAL_RESEARCH"
        }
        return hash(canonical_map.get(self.value, self.value))

# ─────────────────────────────────────────────
# EXPERIMENT 4: PYDANTIC STRUCTURED SCHEMAS
# ─────────────────────────────────────────────

class SourceModel(BaseModel):
    document_name: str = Field(description="Name of the source case document")
    page_number: str = Field(description="Page number in the source document")
    chunk_id: str = Field(description="Unique chunk identifier")
    excerpt: str = Field(description="Relevant text snippet from the document")

class EventType(str, Enum):
    PROSECUTION_ALLEGATION = "Prosecution Allegation"
    WITNESS_OBSERVATION = "Witness Observation"
    CCTV_OBSERVATION = "CCTV Observation"
    FORENSIC_ESTIMATE = "Forensic Estimate"
    DEFENSE_CLAIM = "Defense Claim"
    JUDICIAL_FINDING = "Judicial Finding"
    JUDGMENT_DATE = "Judgment Date"
    CASE_EVENT = "Case Event"

class TimelineEvent(BaseModel):
    event_id: str = Field(description="Unique event identifier (e.g. evt_01)")
    date: Optional[str] = Field(default=None, description="Event date (e.g. 13 November 2024)")
    time: Optional[str] = Field(default=None, description="Specific time or timestamp string (e.g. 21:08 HRS)")
    start_time: Optional[str] = Field(default=None, description="Start time if a range or window (e.g. 21:15 HRS)")
    end_time: Optional[str] = Field(default=None, description="End time if a range or window (e.g. 21:35 HRS)")
    event_description: str = Field(description="Objective factual description of what occurred or was stated")
    actors: List[str] = Field(default_factory=list, description="Persons, entities, or items involved")
    location: Optional[str] = Field(default=None, description="Location of the event if mentioned")
    event_type: str = Field(default=EventType.CASE_EVENT.value, description="Classification of the event")
    source_type: str = Field(default="Case Document", description="Type of source document")
    source_chunk_ids: List[str] = Field(default_factory=list, description="IDs of chunks supporting this event")
    source_document_ids: List[str] = Field(default_factory=list, description="Document names supporting this event")
    supporting_quote: str = Field(default="", description="Exact excerpt from the source chunk proving this event")
    certainty: str = Field(default="Documented", description="Certainty level: Documented, Estimated Window, Contested / Alleged")
    confidence: str = Field(default="High", description="Validation confidence or status")

class TimelineExtractionResult(BaseModel):
    events: List[TimelineEvent] = Field(default_factory=list)
    undated_events: List[TimelineEvent] = Field(default_factory=list)
    selected_chunk_ids: List[str] = Field(default_factory=list)
    selected_document_ids: List[str] = Field(default_factory=list)
    excluded_chunk_ids: List[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    total_chunks_considered: int = 0
    total_events_extracted: int = 0
    validation_status: str = "valid"  # "valid", "partial", "insufficient"
    notes: List[str] = Field(default_factory=list)

class WitnessClaim(BaseModel):
    witness_name: str = Field(description="Name or title of the witness")
    witness_role: str = Field(default="", description="Role or profession of witness")
    statement_source: str = Field(default="", description="Document and page citation")
    source_chunk_id: str = Field(default="", description="Source chunk ID")
    claim_type: str = Field(default="general", description="Subject: timing, location, attire, action, identification")
    claim_text: str = Field(default="", description="The specific factual claim or observation")
    supporting_quote: str = Field(default="", description="Exact quote from the source document")
    time_anchor: Optional[str] = Field(default=None, description="Time mentioned in claim")
    location_anchor: Optional[str] = Field(default=None, description="Location mentioned in claim")
    attire_anchor: Optional[str] = Field(default=None, description="Attire or description mentioned")

class ContradictionComparison(BaseModel):
    comparison_id: str = Field(description="Unique comparison ID (e.g. comp_01)")
    witness_a: str = Field(description="First witness")
    witness_a_claim: str = Field(description="First witness's specific claim")
    witness_a_source: str = Field(description="Source citation for witness A")
    witness_b: str = Field(description="Second witness")
    witness_b_claim: str = Field(description="Second witness's specific claim")
    witness_b_source: str = Field(description="Source citation for witness B")
    factual_subject: str = Field(description="Subject of comparison (e.g. Timing, Location, Attire)")
    nature_of_discrepancy: str = Field(description="Detailed explanation of the discrepancy")
    corroborating_evidence: str = Field(default="", description="Independent corroborating physical or electronic evidence")
    court_resolution: str = Field(default="", description="How the court interpreted or evaluated the discrepancy")
    assessment: str = Field(default="Confirmed Contradiction", description="Confirmed Contradiction, Possible Contradiction / Discrepancy, Compatible Account")
    is_material: bool = Field(default=False, description="Whether the contradiction is material to the case outcome")
    explanation: str = Field(default="", description="Legal explanation of the comparison")

class ContradictionAnalysisResult(BaseModel):
    contradiction_identified: bool = Field(default=False)
    comparisons: List[ContradictionComparison] = Field(default_factory=list)
    witness_claims: List[WitnessClaim] = Field(default_factory=list)
    selected_chunk_ids: List[str] = Field(default_factory=list)
    selected_document_ids: List[str] = Field(default_factory=list)
    physical_evidence_citations: List[str] = Field(default_factory=list)
    court_findings_citations: List[str] = Field(default_factory=list)
    status: str = Field(default="completed")  # "completed", "insufficient_evidence", "no_witnesses_found"
    limitations: List[str] = Field(default_factory=list)

class EvidenceBundle(BaseModel):
    intent: str
    selected_chunk_ids: List[str] = Field(default_factory=list)
    selected_document_ids: List[str] = Field(default_factory=list)
    excluded_chunk_ids: List[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    total_chunks_retrieved: int = 0
    formatted_context: str = ""
    evidence_length_chars: int = 0

class LegalResearchAnswer(BaseModel):
    query: str = Field(description="The user's original legal research question")
    normalized_query: Optional[str] = Field(default=None, description="The preprocessed, normalized research query")
    intent: str = Field(description="Identified research intent category")
    answer: str = Field(description="Detailed grounded legal answer based strictly on retrieved evidence")
    key_findings: List[str] = Field(default_factory=list, description="Key legal facts directly supported by case documents")
    evidence_summary: List[str] = Field(default_factory=list, description="Logical evidence breakdown and witness analysis")
    limitations: List[str] = Field(default_factory=list, description="Information or facts that remain unknown or absent from uploaded documents")
    sources: List[SourceModel] = Field(default_factory=list, description="Verifiable source document references")
    timeline_events: List[TimelineEvent] = Field(default_factory=list, description="Structured timeline events when intent is TIMELINE")
    contradiction_comparisons: List[ContradictionComparison] = Field(default_factory=list, description="Structured witness comparisons when intent is CONTRADICTION")

@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    document_type: str
    page_number: int
    text: str
    case_name: str = "Unknown Case"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "case_name": self.case_name,
            "text": self.text,
            "score": self.score
        }

@dataclass
class ToolExecution:
    tool_name: str
    tool_args: Dict[str, Any]
    status: str  # "completed", "failed"
    result_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status,
            "result_preview": self.result_preview
        }

@dataclass
class WorkflowStep:
    step_name: str
    status: str  # "pending", "in_progress", "completed", "failed"
    detail: str = ""
    tool_execution: Optional[ToolExecution] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step_name": self.step_name,
            "status": self.status,
            "detail": self.detail
        }
        if self.tool_execution:
            d["tool_execution"] = self.tool_execution.to_dict()
        return d

@dataclass
class WorkflowState:
    query: str = ""
    normalized_query: str = ""
    intent: IntentType = IntentType.UNKNOWN
    intent_confidence: float = 1.0
    is_ambiguous: bool = False
    clarification_prompt: str = ""
    normalization_notes: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)
    evidence_strength: str = "Low"  # High, Medium, Low, Insufficient
    response: str = ""
    structured_answer: Optional[LegalResearchAnswer] = None
    raw_model_response: str = ""
    validation_error: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    executed_tools: List[ToolExecution] = field(default_factory=list)
    timeline_events: List[TimelineEvent] = field(default_factory=list)
    evidence_bundle: Optional[EvidenceBundle] = None
    contradiction_result: Optional[ContradictionAnalysisResult] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "Idle"

    def __getattr__(self, name: str) -> Any:
        if name == "metrics":
            self.metrics = {}
            return self.metrics
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value if isinstance(self.intent, IntentType) else self.intent,
            "intent_confidence": self.intent_confidence,
            "is_ambiguous": self.is_ambiguous,
            "clarification_prompt": self.clarification_prompt,
            "normalization_notes": self.normalization_notes,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "evidence_strength": self.evidence_strength,
            "response": self.response,
            "structured_answer": self.structured_answer.model_dump() if self.structured_answer else None,
            "raw_model_response": self.raw_model_response,
            "validation_error": self.validation_error,
            "sources": self.sources,
            "executed_tools": [t.to_dict() for t in self.executed_tools],
            "timeline_events": [e.model_dump() for e in self.timeline_events],
            "evidence_bundle": self.evidence_bundle.model_dump() if self.evidence_bundle else None,
            "contradiction_result": self.contradiction_result.model_dump() if self.contradiction_result else None,
            "metrics": self.metrics,
            "status": self.status
        }

