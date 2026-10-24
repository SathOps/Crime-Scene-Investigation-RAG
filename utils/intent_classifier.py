import re
from typing import Tuple, Dict, List, Optional
from models.schemas import IntentType
from utils.query_normalizer import NormalizedQuery, QueryNormalizer

class IntentClassifier:
    """Robust legal research intent classifier.
    
    Supports:
    - CASE_SUMMARY
    - TIMELINE
    - WITNESS_ANALYSIS
    - CONTRADICTION_DETECTION / CONTRADICTION_ANALYSIS
    - EVIDENCE_EVALUATION / EVIDENCE_ANALYSIS
    - LEGAL_SECTION_CHECK / LEGAL_PROVISIONS
    - GENERAL_RESEARCH / GENERAL_LEGAL_RESEARCH
    - Specialized: ACCUSED_INFORMATION, COURT_FINDINGS, DEFENCE_ARGUMENT, PROSECUTION_ARGUMENT
    """

    INTENT_RULES = {
        IntentType.CONTRADICTION_DETECTION: {
            "weight": 1.3,
            "patterns": [
                r'\b(?:match(?:\s+or\s+not)?|corroborat\w*|clash\w*|differ\w*|discrepan\w*|inconsisten\w*|conflict\w*|contradict\w*)\b',
                r'\bstories\s+(?:match|align|corroborate|differ|contradict)\b',
                r'\bdid\s+(?:their|the)\s+stories\s+match\b',
                r'\b(?:cctv|camera)\s+(?:match|corroborate|conflict|vs|versus)\b',
                r'\b(?:cctv\s+vs\s+guard|camera\s+vs\s+witness)\b',
                r'\bdisagree\w*|refut\w*|disprov\w*|opposit\w*\b',
                r'\balibi\s+(?:false|disproved|untrue|verified)\b'
            ],
            "keywords": ["contradict", "contradiction", "differ", "difference", "discrepancy", "inconsistent", "inconsistency", "match or not", "stories match", "corroborate", "disagree", "disagrees", "disagreement", "vs", "clash"]
        },
        IntentType.TIMELINE: {
            "weight": 1.2,
            "patterns": [
                r'\b(?:timeline|chronolog\w*|sequence(?:\s+of\s+events)?)\b',
                r'\b(?:in\s+order\s+by\s+time|in\s+order|chronologically|by\s+time)\b',
                r'\b(?:when\s+did|at\s+what\s+time|hour\s+by\s+hour)\b',
                r'\bwhat\s+happened\s+first\b',
                r'\bwhat\s+happen(?:ed)?\s+at\s+\d{1,2}:\d{2}\b',
                r'\b\d{1,2}:\d{2}\s*(?:hrs|am|pm)?\b',
                r'\bwhen\s+did\s+(?:he|she|ramesh|vikram)\s+(?:leave|arrive|enter|exit|depart)\b',
                r'\bwhere\s+did\s+(?:he|she|ramesh|the\s+accused)\s+go\b',
                r'\bwhere\s+was\s+he\s+went\b'
            ],
            "keywords": ["timeline", "chronology", "sequence", "when did", "in order", "by time", "dates", "timestamps", "order of events", "what happen at", "where did he go"]
        },
        IntentType.WITNESS_ANALYSIS: {
            "weight": 1.0,
            "patterns": [
                r'\bwhat\s+did\s+(?:the\s+)?(?:guard|witness|ankit|priya|suresh)\s+(?:say|state|claim|depose)\b',
                r'\bwhat\s+does\s+(?:the\s+)?witness\s+say\b',
                r'\b(?:witness|eyewitness)\s+statement\w*\b',
                r'\b(?:deposition|testimony)\b'
            ],
            "keywords": ["witness", "witnesses", "guard say", "priya say", "ankit say", "suresh say", "statement", "deposition", "testimony"]
        },
        IntentType.CASE_SUMMARY: {
            "weight": 1.1,
            "patterns": [
                r'\b(?:sum\s+up|summariz\w*|overview|synopsis|brief|tldr|tl;dr)\b',
                r'\bwhat\s+happened\s+in\s+this\s+case\b',
                r'\bgive\s+me\s+the\s+tldr\b',
                r'\b(?:whole|entire|complete)\s+case\b',
                r'\bcase\s+summary\b'
            ],
            "keywords": ["summary", "summarize", "sum up", "overview", "what happened in this case", "whole case", "brief overview", "tldr"]
        },
        IntentType.EVIDENCE_EVALUATION: {
            "weight": 1.1,
            "patterns": [
                r'\b(?:evidence|forensic|ballistic|fingerprint|blood|dna)\b',
                r'\b(?:9mm|shell\s+casing|casing|wristwatch|briefcase)\b',
                r'\b(?:gun|weapon|pistol|firearm|revolver)\b',
                r'\b(?:cctv\s+footage|camera\s+recording|cdr|call\s+detail\s+records?|cell\s+tower)\b',
                r'\bwhat\s+evidence\s+connects\b',
                r'\bdid\s+they\s+see\s+the\s+gun\b',
                r'\bpostmortem|autopsy\b'
            ],
            "keywords": ["evidence", "forensic", "weapon", "gun", "blood", "fingerprint", "casing", "wristwatch", "briefcase", "cctv", "autopsy", "postmortem", "cdr", "cell tower"]
        },
        IntentType.LEGAL_SECTION_CHECK: {
            "weight": 1.2,
            "patterns": [
                r'\b(?:section|sec\.?)\s*(?:302|392|120b|34|201|164)\b',
                r'\b(?:ipc|crpc|penal\s+code|bns)\b',
                r'\bwhat\s+sections?\s+(?:are\s+)?(?:charged|applied|invoked)\b',
                r'\b(?:legal\s+provisions?|charges?\s+framed|statute|act)\b'
            ],
            "keywords": ["section", "ipc", "crpc", "statute", "act", "provision", "charges", "law applied", "penal code"]
        },
        IntentType.ACCUSED_INFORMATION: {
            "weight": 1.0,
            "patterns": [
                r'\b(?:who\s+is\s+the\s+accused|accused\s+details|about\s+ramesh\s+kumar)\b',
                r'\b(?:suspect|defendant|alibi\s+claim|alibi\s+at)\b'
            ],
            "keywords": ["accused", "suspect", "defendant", "alibi", "alibi claim", "blue dart cafe"]
        },
        IntentType.COURT_FINDINGS: {
            "weight": 0.9,
            "patterns": [
                r'\b(?:court\s+judgment|ruling|verdict|order\s+of\s+the\s+court|judge\s+findings)\b'
            ],
            "keywords": ["court", "judge", "ruling", "verdict", "judgment"]
        }
    }

    @classmethod
    def classify(cls, query: str, norm_obj: Optional[NormalizedQuery] = None) -> Tuple[IntentType, float]:
        """Classifies intent using normalized and original query signals."""
        if not norm_obj:
            norm_obj = QueryNormalizer.normalize(query)

        # Ambiguous query check
        if norm_obj.is_ambiguous:
            return IntentType.UNKNOWN, 0.0

        # Explicit court-outcome or forensic-results requests take precedence over
        # incidental words such as "contradictions" in a compound query.
        query_text = (query + " " + norm_obj.normalized_query).lower()
        asks_court_outcome = any(term in query_text for term in (
            "court judgment", "court judgement", "final court judgment", "final order",
            "court findings", "judge findings", "verdict", "acquitted", "convicted"
        ))
        asks_forensic_results = any(term in query_text for term in (
            "forensic findings", "forensic report", "laboratory findings",
            "forensic results", "ballistic findings", "fingerprint analysis"
        ))
        if asks_court_outcome:
            return IntentType.COURT_FINDINGS, 0.97
        if asks_forensic_results:
            return IntentType.EVIDENCE_EVALUATION, 0.95

        # Canonical hint match
        if norm_obj.inferred_intent_hint:
            hint = norm_obj.inferred_intent_hint
            if hint == "CONTRADICTION_DETECTION":
                return IntentType.CONTRADICTION_DETECTION, 0.98
            elif hint == "TIMELINE":
                return IntentType.TIMELINE, 0.98
            elif hint == "WITNESS_ANALYSIS":
                return IntentType.WITNESS_ANALYSIS, 0.98
            elif hint == "CASE_SUMMARY":
                return IntentType.CASE_SUMMARY, 0.98
            elif hint == "LEGAL_SECTION_CHECK":
                return IntentType.LEGAL_SECTION_CHECK, 0.98
            elif hint == "EVIDENCE_EVALUATION":
                return IntentType.EVIDENCE_EVALUATION, 0.98

        norm_text = norm_obj.normalized_query.lower()
        orig_text = query.lower()

        scores: Dict[IntentType, float] = {}

        for intent, rules in cls.INTENT_RULES.items():
            score = 0.0
            weight = rules.get("weight", 1.0)

            # Regex patterns
            for pat in rules.get("patterns", []):
                if re.search(pat, norm_text, re.I):
                    score += 2.8
                elif re.search(pat, orig_text, re.I):
                    score += 2.2

            # Keywords
            for kw in rules.get("keywords", []):
                if kw in norm_text:
                    score += 1.4
                elif kw in orig_text:
                    score += 1.0

            scores[intent] = score * weight

        # Special priority handling:
        # If query has contradiction/comparison markers regarding witnesses or CCTV:
        has_contradiction_word = any(w in norm_text or w in orig_text for w in ["differ", "difference", "discrepanc", "disagree", "clash", "conflict", "match", "vs", "versus", "corroborat", "inconsisten"])
        if ("witness" in norm_text or "guard" in norm_text or "priya" in norm_text or "ankit" in norm_text) and has_contradiction_word:
            scores[IntentType.CONTRADICTION_DETECTION] = scores.get(IntentType.CONTRADICTION_DETECTION, 0) + 4.5

        if ("cctv" in norm_text or "camera" in norm_text) and has_contradiction_word:
            scores[IntentType.CONTRADICTION_DETECTION] = scores.get(IntentType.CONTRADICTION_DETECTION, 0) + 4.0

        # Special: "tldr" or "sum up"
        if "tldr" in orig_text or "sum up" in orig_text:
            scores[IntentType.CASE_SUMMARY] = scores.get(IntentType.CASE_SUMMARY, 0) + 4.0

        # Special: "gun" or "weapon" without legal section -> EVIDENCE_EVALUATION
        if any(w in orig_text for w in ["gun", "weapon", "casing", "call detail record", "cdr"]):
            scores[IntentType.EVIDENCE_EVALUATION] = scores.get(IntentType.EVIDENCE_EVALUATION, 0) + 3.5

        # Special: "where was he went" or "what happen at 21:25" -> TIMELINE
        if "where" in orig_text and ("go" in norm_text or "went" in orig_text):
            scores[IntentType.TIMELINE] = scores.get(IntentType.TIMELINE, 0) + 3.5
        if re.search(r'\b\d{1,2}:\d{2}\b', orig_text):
            scores[IntentType.TIMELINE] = scores.get(IntentType.TIMELINE, 0) + 3.0

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score < 1.0:
            return IntentType.GENERAL_RESEARCH, 0.3

        confidence = min(0.99, round(best_score / (best_score + 1.5), 2))
        return best_intent, confidence
