import re
import string
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

@dataclass
class NormalizedQuery:
    original_query: str
    normalized_query: str
    is_ambiguous: bool = False
    clarification_prompt: str = ""
    entities_preserved: List[str] = field(default_factory=list)
    legal_sections_preserved: List[str] = field(default_factory=list)
    dates_times_preserved: List[str] = field(default_factory=list)
    normalization_notes: List[str] = field(default_factory=list)
    inferred_intent_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "is_ambiguous": self.is_ambiguous,
            "clarification_prompt": self.clarification_prompt,
            "entities_preserved": self.entities_preserved,
            "legal_sections_preserved": self.legal_sections_preserved,
            "dates_times_preserved": self.dates_times_preserved,
            "normalization_notes": self.normalization_notes,
            "inferred_intent_hint": self.inferred_intent_hint
        }


class QueryNormalizer:
    """Preprocesses and normalizes raw user queries.
    
    Handles:
    - Spelling mistakes and phonetic approximations.
    - Shorthand abbreviations, slang, and informal syntax.
    - Grammar corrections (e.g., tense disagreements).
    - Preservation of critical legal entities, names, dates, times, and sections.
    - Ambiguity detection and clarifying question generation.
    - Zero query answering during normalization.
    """

    # Domain Named Entities (Preserved with strict casing)
    KNOWN_PERSONS = [
        ("ramesh kumar", "Ramesh Kumar"),
        ("vikram malhotra", "Vikram Malhotra"),
        ("ankit sharma", "Ankit Sharma"),
        ("priya nair", "Priya Nair"),
        ("suresh verma", "Suresh Verma"),
        ("rajesh kumar", "Rajesh Kumar"),
        ("dr. a. k. deshmukh", "Dr. A. K. Deshmukh"),
        ("insp. s. r. patil", "Insp. S. R. Patil"),
        ("ramesh", "Ramesh Kumar"),
        ("vikram", "Vikram Malhotra"),
        ("malhotra", "Vikram Malhotra"),
        ("ankit", "Ankit Sharma"),
        ("priya", "Priya Nair"),
        ("suresh", "Suresh Verma")
    ]

    KNOWN_LOCATIONS = [
        ("apex towers", "Apex Towers"),
        ("apex logistics", "Apex Logistics"),
        ("office #402", "Office #402"),
        ("office 402", "Office #402"),
        ("blue dart cafe", "Blue Dart Cafe"),
        ("shivajinagar", "Shivajinagar"),
        ("aundh", "Aundh"),
        ("kothrud", "Kothrud"),
        ("pune", "Pune")
    ]

    LEGAL_SECTIONS_PATTERNS = [
        (r'\b(?:section|sec\.?)\s*(302|392|120b|120-b|34|201)\b(?:\s*ipc)?', r'Section \1 IPC'),
        (r'\bipc\s*(302|392|120b|120-b|34|201)\b', r'Section \1 IPC'),
        (r'\bfir\s*(?:no\.?|number)?\s*(142/2024|142)\b', r'FIR No. 142/2024'),
        (r'\bcrpc\s*(?:section|sec\.?)?\s*(\d+)\b', r'CrPC Section \1')
    ]

    EVIDENCE_ITEMS = [
        ("item-a", "Evidence Item-A (9mm spent shell casing)"),
        ("item-b", "Evidence Item-B (fired 9mm bullet slug)"),
        ("item-c", "Evidence Item-C (bloodstained wristwatch)"),
        ("item-d", "Evidence Item-D (latent fingerprint)"),
        ("item-e", "Evidence Item-E (stolen financial audit briefcase)"),
        ("camera 01", "CCTV Camera 01 (Rear gate)"),
        ("camera 03", "CCTV Camera 03 (4th floor corridor)"),
        ("cam 01", "CCTV Camera 01"),
        ("cam 03", "CCTV Camera 03")
    ]

    # Shorthand & Typo mappings
    ABBREVIATIONS = {
        "tldr": "summary",
        "tl;dr": "summary",
        "wht": "what",
        "wat": "what",
        "wen": "when",
        "wn": "when",
        "wr": "where",
        "hw": "how",
        "whi": "why",
        "y": "why",
        "n": "and",
        "nd": "and",
        "&": "and",
        "bcuz": "because",
        "bcoz": "because",
        "cuz": "because",
        "coz": "because",
        "abt": "about",
        "plz": "please",
        "pls": "please",
        "witnes": "witness",
        "wittness": "witness",
        "witnss": "witness",
        "wits": "witness",
        "witneses": "witnesses",
        "statemnt": "statement",
        "statemnts": "statements",
        "statmnt": "statement",
        "statmnts": "statements",
        "statment": "statement",
        "statments": "statements",
        "stmts": "statements",
        "stmt": "statement",
        "stms": "statements",
        "diffrnce": "difference",
        "diffrnc": "difference",
        "diffrence": "difference",
        "differnce": "difference",
        "diference": "difference",
        "diff": "difference",
        "evrything": "everything",
        "evrythng": "everything",
        "everthing": "everything",
        "happend": "happened",
        "hapened": "happened",
        "hpened": "happened",
        "camer": "camera",
        "cams": "cameras",
        "cam": "camera",
        "alibi": "alibi",
        "aliby": "alibi",
        "acused": "accused",
        "acusedd": "accused",
        "suspet": "suspect",
        "vicim": "victim",
        "victm": "victim",
        "weapn": "weapon",
        "gunn": "gun",
        "pm report": "postmortem report",
        "autopsi": "autopsy",
        "postmortm": "postmortem",
        "forensik": "forensic",
        "forensix": "forensic",
        "timline": "timeline",
        "chronolog": "chronology",
        "securty": "security",
        "secuity": "security",
        "gurd": "guard",
        "disagrees": "disagree",
        "inconsistenci": "inconsistency",
        "inconsistencys": "inconsistencies",
        "discrepensi": "discrepancy",
        "discrepancies": "discrepancies",
        "cdr": "Call Detail Records (CDR)",
        "cctv": "CCTV footage"
    }

    # Common grammatical fixes
    GRAMMAR_FIXES = [
        (r'\bwhen did he left\b', 'when did he leave'),
        (r'\bdid he left\b', 'did he leave'),
        (r'\bdid they saw\b', 'did they see'),
        (r'\bdid he went\b', 'did he go'),
        (r'\bwhere was he went\b', 'where did he go'),
        (r'\bwhat happen at\b', 'what happened at'),
        (r'\bwhat happen\b', 'what happened'),
        (r'\bwhy does witness statements disagrees\b', 'why do the witness statements disagree'),
        (r'\bwhy does witness statements disagree\b', 'why do the witness statements disagree'),
        (r'\bwho done it\b', 'who committed the crime'),
        (r'\bdoes cctv match\b', 'does the CCTV footage corroborate'),
        (r'\bdo cctv match\b', 'does the CCTV footage corroborate')
    ]

    # Idiomatic paraphrases & high-impact domain sentence mappings
    CANONICAL_PHRASE_MAPPINGS = [
        (
            re.compile(r'^(?:wht|what)\s+did\s+(?:the\s+)?guard\s+say\s+(?:n|and)\s+does\s+(?:cctv|camera)\s+match\s+it\??$', re.I),
            "What did security guard Ankit Sharma state, and does the CCTV footage corroborate his statement?",
            "CONTRADICTION_DETECTION"
        ),
        (
            re.compile(r'^(?:any\s+)?(?:diffrnce|difference|diff|discrepancy|inconsistency)\s+in\s+(?:witnes|witness)\s+(?:statemnts|statements|stmts)\s+(?:n|and)\s+(?:camera|cctv)\??$', re.I),
            "What discrepancies or differences exist between the witness statements and the CCTV camera footage?",
            "CONTRADICTION_DETECTION"
        ),
        (
            re.compile(r'^(?:cctv|camera)\s+vs\s+(?:guard|witness)\s+statements?\??$', re.I),
            "Compare the CCTV footage against the security guard's witness statements for contradictions or discrepancies.",
            "CONTRADICTION_DETECTION"
        ),
        (
            re.compile(r'^(?:when\s+did\s+he\s+(?:left|leave)\s+(?:the\s+place|the\s+building|apex\s+towers)?\??)$', re.I),
            "When did the accused, Ramesh Kumar, leave the crime scene at Apex Towers?",
            "TIMELINE"
        ),
        (
            re.compile(r'^(?:show\s+me\s+)?(?:evrything|everything)\s+that\s+(?:happend|happened)\s+in\s+order\??$', re.I),
            "Provide a chronological timeline of all events that occurred in this case in sequential order.",
            "TIMELINE"
        ),
        (
            re.compile(r'^(?:put|list|arrange)\s+(?:everything|all\s+events)\s+(?:in\s+order\s+by\s+time|by\s+time|chronologically)\??$', re.I),
            "Provide a chronological timeline of all case events in sequential order.",
            "TIMELINE"
        ),
        (
            re.compile(r'^(?:did\s+|do\s+)?their\s+stories\s+match(?:\s+or\s+not)?\??$', re.I),
            "Do the witness statements corroborate each other, or are there material contradictions between them?",
            "CONTRADICTION_DETECTION"
        ),
        (
            re.compile(r'^(?:sum\s+up|summarize|give\s+a\s+summary\s+of|give\s+me\s+the\s+tldr\s+of)\s+(?:the\s+whole\s+case|the\s+case|all\s+facts|what\s+happened)\??$', re.I),
            "Provide a comprehensive summary of the case facts, charges, and investigation findings.",
            "CASE_SUMMARY"
        ),
        (
            re.compile(r'^(?:what\s+does|what\s+did)\s+(?:the\s+)?witness(?:es)?\s+say\??$', re.I),
            "What did the key witnesses state in their police depositions regarding the incident?",
            "WITNESS_ANALYSIS"
        ),
        (
            re.compile(r'^(?:what\s+sections?\s+(?:are\s+)?(?:charged|applied|invoked))\??$', re.I),
            "What legal provisions and IPC sections are invoked against the accused in the FIR and chargesheet?",
            "LEGAL_SECTION_CHECK"
        ),
        (
            re.compile(r'^(?:what\s+evidence\s+connects\s+(?:him|the\s+accused|ramesh)\s+to\s+(?:the\s+crime|murder))\??$', re.I),
            "What physical, forensic, and electronic evidence connects the accused, Ramesh Kumar, to the murder?",
            "EVIDENCE_EVALUATION"
        )
    ]

    AMBIGUOUS_PATTERNS = [
        re.compile(r'^(?:what\s+about\s+that|did\s+it|tell\s+me\s+more|status|more\s+info|what\s+next|and\s+then)\??$', re.I),
        re.compile(r'^(?:who|what|why|when|where|how)\??$', re.I),
        re.compile(r'^(?:tell\s+me\s+about\s+him|tell\s+me\s+about\s+her)\??$', re.I)
    ]

    @classmethod
    def normalize(cls, query: str) -> NormalizedQuery:
        """Main entry point for intelligent query preprocessing."""
        notes = []
        raw = (query or "").strip()
        if not raw:
            return NormalizedQuery(
                original_query=query,
                normalized_query="",
                is_ambiguous=True,
                clarification_prompt="Please provide a specific case question or legal research topic."
            )

        # 1. Check for Ambiguity
        for ambig_pat in cls.AMBIGUOUS_PATTERNS:
            if ambig_pat.match(raw) or len(raw.split()) < 2:
                notes.append("Detected ambiguous or under-specified input.")
                return NormalizedQuery(
                    original_query=query,
                    normalized_query=raw,
                    is_ambiguous=True,
                    clarification_prompt=(
                        "Your question is ambiguous. Could you please specify which person "
                        "(e.g., Ramesh Kumar, Vikram Malhotra, Ankit Sharma, Priya Nair), "
                        "evidence (e.g., CCTV footage, 9mm casing, CDR, wristwatch), "
                        "or legal issue (e.g., FIR charges, alibi claim, timeline) you are asking about?"
                    ),
                    normalization_notes=notes
                )

        # 2. Check Canonical Phrase Mappings for instant high-precision match
        for pat, canonical_form, intent_hint in cls.CANONICAL_PHRASE_MAPPINGS:
            if pat.search(raw):
                notes.append(f"Matched canonical legal inquiry pattern -> Inferred intent: {intent_hint}")
                # Extract entities from canonical
                entities = cls._extract_entities(canonical_form)
                sections = cls._extract_legal_sections(canonical_form)
                times = cls._extract_dates_times(canonical_form)
                return NormalizedQuery(
                    original_query=query,
                    normalized_query=canonical_form,
                    is_ambiguous=False,
                    entities_preserved=entities,
                    legal_sections_preserved=sections,
                    dates_times_preserved=times,
                    normalization_notes=notes,
                    inferred_intent_hint=intent_hint
                )

        # 3. Dynamic Multi-Stage Normalization
        text = raw

        # Step A: Grammar transformations
        for g_pat, g_repl in cls.GRAMMAR_FIXES:
            new_text = re.sub(g_pat, g_repl, text, flags=re.I)
            if new_text != text:
                notes.append(f"Applied grammar correction: '{g_pat}' -> '{g_repl}'")
                text = new_text

        # Step B: Protect Legal Sections & Formats
        sections_found = []
        for s_pat, s_repl in cls.LEGAL_SECTIONS_PATTERNS:
            matches = re.findall(s_pat, text, flags=re.I)
            if matches:
                sections_found.extend([f"Section {m} IPC" if not str(m).startswith("FIR") else m for m in matches])
                text = re.sub(s_pat, s_repl, text, flags=re.I)

        # Step C: Token-level Abbreviation & Typo correction
        words = text.split()
        normalized_words = []
        for word in words:
            # Strip punctuation for lookup
            clean_word = word.lower().strip(string.punctuation)
            if clean_word in cls.ABBREVIATIONS:
                replacement = cls.ABBREVIATIONS[clean_word]
                # Preserve attached punctuation
                prefix = ""
                suffix = ""
                if word and word[0] in string.punctuation:
                    prefix = word[0]
                if word and word[-1] in string.punctuation:
                    suffix = word[-1]
                normalized_words.append(f"{prefix}{replacement}{suffix}")
                notes.append(f"Expanded abbreviation/typo '{clean_word}' -> '{replacement}'")
            else:
                # Fuzzy match close domain typos if length >= 4
                fuzzy_rep = cls._fuzzy_match_domain_words(clean_word)
                if fuzzy_rep:
                    prefix = word[0] if word and word[0] in string.punctuation else ""
                    suffix = word[-1] if word and word[-1] in string.punctuation else ""
                    normalized_words.append(f"{prefix}{fuzzy_rep}{suffix}")
                    notes.append(f"Fuzzy corrected '{clean_word}' -> '{fuzzy_rep}'")
                else:
                    normalized_words.append(word)

        text = " ".join(normalized_words)

        # Step D: Preserve & Capitalize Known Named Entities
        entities_found = []
        for low_ent, proper_ent in cls.KNOWN_PERSONS + cls.KNOWN_LOCATIONS:
            pattern = r'\b' + re.escape(low_ent) + r'\b'
            if re.search(pattern, text, re.I):
                text = re.sub(pattern, proper_ent, text, flags=re.I)
                entities_found.append(proper_ent)

        # Evidence Items
        for ev_low, ev_proper in cls.EVIDENCE_ITEMS:
            pattern = r'\b' + re.escape(ev_low) + r'\b'
            if re.search(pattern, text, re.I):
                text = re.sub(pattern, ev_proper, text, flags=re.I)
                entities_found.append(ev_proper)

        # Step E: Extract dates & times
        dates_times_found = cls._extract_dates_times(text)

        # Remove adjacent duplicate tokens introduced by abbreviation/entity corrections.
        text = re.sub(r"\b([\\w#-]+)(\s+\1\b)+", r"\1", text, flags=re.I)

        # Step F: Ensure proper casing & punctuation
        text = text.strip()
        if text:
            # Capitalize first character
            text = text[0].upper() + text[1:]
            # Ensure ending punctuation
            if not text.endswith(('?', '.', '!')):
                if any(text.lower().startswith(q) for q in ["what", "when", "where", "who", "why", "how", "did", "does", "is", "are", "can", "do"]):
                    text += "?"
                else:
                    text += "."

        # Final cleanup prevents aliases creating "Ramesh Kumar Kumar" or "CCTV footage footage".
        text = re.sub(r"\b([\\w#-]+)(\s+\1\b)+", r"\1", text, flags=re.I)

        # Deduplicate entities
        entities_found = sorted(list(set(entities_found)))
        sections_found = sorted(list(set(sections_found)))

        return NormalizedQuery(
            original_query=query,
            normalized_query=text,
            is_ambiguous=False,
            entities_preserved=entities_found,
            legal_sections_preserved=sections_found,
            dates_times_preserved=dates_times_found,
            normalization_notes=notes
        )

    @classmethod
    def _fuzzy_match_domain_words(cls, word: str) -> Optional[str]:
        """Simple Levenshtein distance matching for domain-critical terms."""
        if len(word) < 4:
            return None
        
        candidates = {
            "witness": ["witnes", "wittnes", "wits", "witnss"],
            "witnesses": ["witneses", "wittnesses", "witnesess"],
            "statement": ["statmnt", "statemnt", "statment"],
            "statements": ["statemnts", "statmnts", "statments", "stmts"],
            "difference": ["diffrnce", "diffrence", "differnce", "diference"],
            "discrepancy": ["discrepensi", "discrepency", "discrepenci"],
            "inconsistency": ["inconsistenci", "inconsistensy"],
            "everything": ["evrything", "evrythng", "everthing"],
            "happened": ["happend", "hapened", "hpened"],
            "timeline": ["timline", "timeln"],
            "corroborate": ["coroborate", "coroborate"],
            "contradiction": ["contradiction", "contradictin"],
            "casing": ["casng", "caasing"],
            "forensic": ["forensik", "forensix"]
        }

        for canonical, misspellings in candidates.items():
            if word in misspellings:
                return canonical
            # Distance 1 check
            for m in misspellings:
                if cls._levenshtein_distance(word, m) <= 1:
                    return canonical

        return None

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return QueryNormalizer._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    @classmethod
    def _extract_entities(cls, text: str) -> List[str]:
        found = []
        for _, proper in cls.KNOWN_PERSONS + cls.KNOWN_LOCATIONS:
            if proper.lower() in text.lower():
                found.append(proper)
        return sorted(list(set(found)))

    @classmethod
    def _extract_legal_sections(cls, text: str) -> List[str]:
        sections = []
        for pat, _ in cls.LEGAL_SECTIONS_PATTERNS:
            matches = re.findall(pat, text, flags=re.I)
            if matches:
                sections.extend([f"Section {m} IPC" if not str(m).startswith("FIR") else m for m in matches])
        return sorted(list(set(sections)))

    @classmethod
    def _extract_dates_times(cls, text: str) -> List[str]:
        pat = r'(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|\b\d{1,2}:\d{2}\s*(?:HRS|AM|PM)?\b)'
        matches = re.findall(pat, text, flags=re.I)
        return sorted(list(set(matches)))
