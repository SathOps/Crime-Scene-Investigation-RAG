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

class LegalResearchAnswer(BaseModel):
    query: str = Field(description="The user's original legal research question")
    normalized_query: Optional[str] = Field(default=None, description="The preprocessed, normalized research query")
    intent: str = Field(description="Identified research intent category")
    answer: str = Field(description="Detailed grounded legal answer based strictly on retrieved evidence")
    key_findings: List[str] = Field(default_factory=list, description="Key legal facts directly supported by case documents")
    evidence_summary: List[str] = Field(default_factory=list, description="Logical evidence breakdown and witness analysis")
    limitations: List[str] = Field(default_factory=list, description="Information or facts that remain unknown or absent from uploaded documents")
    sources: List[SourceModel] = Field(default_factory=list, description="Verifiable source document references")

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
    status: str = "Idle"

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
            "status": self.status
        }

