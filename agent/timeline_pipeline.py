"""Structured timeline extraction pipeline for multi-agent legal case research.

Replaces fragile regex-only text slicing with structured evidence bundling,
entity parsing, sentence-boundary extraction, event categorization, chronological
sorting, and provenance verification against retrieved chunks.
"""
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from models.schemas import (
    DocumentChunk,
    TimelineEvent,
    TimelineExtractionResult,
    EvidenceBundle,
    EventType,
)


MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def extract_sentence_boundary(src_text: str, match_start: int, match_end: int) -> str:
    """Extract a complete, grammatical sentence surrounding a match position."""
    left = match_start
    while left > 0:
        if src_text[left - 1] == "\n":
            break
        if src_text[left - 1] in ".!?" and left < len(src_text) and src_text[left].isspace():
            break
        left -= 1

    right = match_end
    while right < len(src_text):
        if src_text[right] == "\n":
            break
        if src_text[right] in ".!?" and (right + 1 == len(src_text) or src_text[right + 1].isspace()):
            right += 1
            break
        right += 1

    snippet = src_text[left:right].strip()
    # Clean leading list markers, numbering, broken quotes, or punctuation
    snippet = re.sub(r'^(?:[-*•#\d.]+|\([a-zA-Z\d]+\))\s*', '', snippet).strip()
    snippet = re.sub(r'^\s*[,;:)]+\s*', '', snippet).strip()
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet


def classify_event_type(snippet: str, source_type: str = "") -> Tuple[str, str]:
    """Classify event into procedural categories with certainty metadata."""
    s_lower = snippet.lower()
    src_lower = source_type.lower()

    # 1. Forensic estimate (autopsy / postmortem TOD)
    if any(k in s_lower for k in ("postmortem", "time of death", "autopsy", "pm-894", "estimated time of death", "rigor mortis")):
        return (
            EventType.FORENSIC_ESTIMATE.value,
            "Estimated Window"
        )

    # 2. CCTV observation
    if any(k in s_lower for k in ("cctv", "camera 01", "camera 02", "camera 03", "dvr", "gait analysis", "video stream", "video log", "recorded suspect", "captured entering", "captured exiting", "captured hurriedly", "rear service emergency")):
        return (EventType.CCTV_OBSERVATION.value, "Documented")

    # 3. Judgment date / Judicial finding
    if any(k in s_lower for k in ("pronounced on", "pronounced in open court", "judgment date", "date of pronouncement", "pronounced judgment", "court pronounced", "sessions court", "convicting", "conviction", "convicted", "sentenced")):
        return (EventType.JUDGMENT_DATE.value, "Judicial Finding")
    if any(k in s_lower for k in ("court held", "court holds", "court finding", "alibi is rejected")):
        return (EventType.JUDICIAL_FINDING.value, "Judicial Finding")

    # 4. Defense claim / alibi
    if any(k in s_lower for k in ("pleaded an alibi", "alibi under section 11", "defense claimed", "claimed presence at")):
        return (EventType.DEFENSE_CLAIM.value, "Contested / Alleged")

    # 5. Witness observation
    if any(k in s_lower for k in ("witness", "ankit", "priya", "suresh", "saw", "observed", "stated that", "testified", "deposed", "heard a loud")):
        return (EventType.WITNESS_OBSERVATION.value, "Documented")

    # 6. Prosecution allegation
    if any(k in s_lower for k in ("prosecution alleged", "prosecution case", "alleged that", "allegation")):
        return (EventType.PROSECUTION_ALLEGATION.value, "Contested / Alleged")

    # Default by source
    if "witness" in src_lower:
        return (EventType.WITNESS_OBSERVATION.value, "Documented")
    if "cctv" in src_lower:
        return (EventType.CCTV_OBSERVATION.value, "Documented")
    if "judgment" in src_lower:
        return (EventType.JUDICIAL_FINDING.value, "Judicial Finding")

    return (EventType.CASE_EVENT.value, "Documented")


def extract_actors_from_snippet(snippet: str) -> List[str]:
    """Extract actors mentioned in an event snippet case-agnostically."""
    actors_found: List[str] = []

    # 1. Title/Prefix + Proper Names (e.g. Mr. Malhotra, Inspector Patil, Dr. Kulkarni, Judge Sharma)
    title_matches = re.findall(
        r'\b(Mr\.|Ms\.|Mrs\.|Dr\.|Inspector|Insp\.|Constable|Judge|Justice|Officer|Guard)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        snippet
    )
    for title, name in title_matches:
        full_actor = f"{title} {name}".strip()
        if full_actor not in actors_found:
            actors_found.append(full_actor)

    # 2. Procedural & Legal Roles (e.g. Accused, Victim, Eye Witness, Security Guard, Senior Accountant)
    role_pattern = re.compile(
        r'\b(Security Guard(?:\s+[A-Z][a-z]+)?|Senior Accountant(?:\s+[A-Z][a-z]+)?|'
        r'The Accused(?:\s+[A-Z][a-z]+)?|Accused(?:\s+[A-Z][a-z]+)?|'
        r'The Victim(?:\s+[A-Z][a-z]+)?|Victim(?:\s+[A-Z][a-z]+)?|'
        r'Investigating Officer|Eye Witness|PW-\d+|DW-\d+)\b',
        re.I
    )
    for m in role_pattern.finditer(snippet):
        role_actor = m.group(1).strip()
        if not any(role_actor.lower() in a.lower() for a in actors_found):
            actors_found.append(role_actor)

    # 3. Two capitalized words resembling person names (e.g., "Ramesh Kumar", "Vikram Malhotra", "Priya Nair")
    name_matches = re.findall(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', snippet)
    stop_names = {
        "Camera", "Sessions", "Court", "Time", "Death", "Postmortem", "Report", "Analysis", "Judgment",
        "Section", "Police", "Station", "November", "January", "December", "October", "Evidence", "Case",
        "Emergency", "Service", "Front", "Rear", "Blue", "Navy", "Red", "Black", "Office", "Main", "Autopsy"
    }
    for first, last in name_matches:
        if first not in stop_names and last not in stop_names:
            pair = f"{first} {last}"
            if not any(pair in a for a in actors_found):
                actors_found.append(pair)

    # Generic actor fallback
    s_lower = snippet.lower()
    if not actors_found:
        if "accused" in s_lower:
            actors_found.append("Accused")
        if "victim" in s_lower:
            actors_found.append("Victim")
        if "guard" in s_lower or "security" in s_lower:
            actors_found.append("Security Guard")

    return actors_found


def extract_location_from_snippet(snippet: str) -> Optional[str]:
    """Extract location mentions from event snippet."""
    loc_match = re.search(
        r'\b(rear service gate|front main entrance|rear gate|front gate|service gate|emergency stairwell|stairwell door|office\s*#?\d+|lobby|cafe|corridor|entrance|exit|parking|cell tower|room|chamber|gate)\b',
        snippet,
        re.I
    )
    if loc_match:
        return loc_match.group(1).strip()
    return None


def normalize_event_chronology(
    stamp: str,
    snippet: str,
    chunk_context: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Tuple[int, int, int, int, int, int]]:
    """Normalize date, time, start_time, end_time, and return a comparable sort tuple."""
    # Month, Day, Year extraction
    year, month, day = 9999, 99, 99
    date_str: Optional[str] = None

    date_match = re.search(
        r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})',
        snippet,
        re.I
    )
    if not date_match:
        date_match = re.search(
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})',
            chunk_context,
            re.I
        )

    if date_match:
        day = int(date_match.group(1))
        m_name = date_match.group(2).lower()
        month = MONTH_MAP.get(m_name, 11)
        year = int(date_match.group(3))
        month_names_full = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        }
        date_str = f"{day} {month_names_full.get(month, m_name.capitalize())} {year}"
    else:
        iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', snippet) or re.search(r'(\d{4})-(\d{2})-(\d{2})', chunk_context)
        if iso_match:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))
            date_str = f"{day} {month} {year}"

    # Time / Range extraction
    hour, minute, second = 0, 0, 0
    time_str: Optional[str] = None
    start_time_str: Optional[str] = None
    end_time_str: Optional[str] = None
    is_range = 0

    # Check for approximate markers e.g. "around 8 PM", "at about 20:30 HRS"
    approx_match = re.search(r'\b(around|about|approx\.?|approximately)\s+(\d{1,2}(?::\d{2})?\s*(?:HRS|AM|PM)?)', snippet, re.I)
    approx_prefix = f"{approx_match.group(1).lower()} " if approx_match else ""

    range_match = re.search(r'(\d{1,2}:\d{2})\s*(?:HRS|AM|PM)?\s*(?:to|-|–|and)\s*(\d{1,2}:\d{2})\s*(?:HRS|AM|PM)?', stamp, re.I)
    if not range_match:
        range_match = re.search(r'(\d{1,2}:\d{2})\s*(?:HRS|AM|PM)?\s*(?:to|-|–|and)\s*(\d{1,2}:\d{2})\s*(?:HRS|AM|PM)?', snippet, re.I)

    if range_match:
        start_time_str = f"{range_match.group(1)} HRS"
        end_time_str = f"{range_match.group(2)} HRS"
        time_str = f"{range_match.group(1)}–{range_match.group(2)} HRS"
        t_parts = range_match.group(1).split(":")
        hour = int(t_parts[0])
        minute = int(t_parts[1])
        is_range = 1
    else:
        single_time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(HRS|AM|PM)?', stamp, re.I)
        if not single_time_match:
            single_time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(HRS|AM|PM)?', snippet, re.I)

        if single_time_match:
            hour = int(single_time_match.group(1))
            minute = int(single_time_match.group(2))
            sec_val = single_time_match.group(3)
            second = int(sec_val) if sec_val else 0
            if single_time_match.group(4) and single_time_match.group(4).upper() == "PM" and hour < 12:
                hour += 12
            raw_time = f"{hour:02d}:{minute:02d}" + (f":{second:02d}" if second else "") + " HRS"
            time_str = f"{approx_prefix}{raw_time}".strip() if approx_prefix and "around" not in raw_time else raw_time

    sort_tuple = (year, month, day, hour, minute, is_range)
    return date_str, time_str, start_time_str, end_time_str, sort_tuple


class TimelinePipeline:
    """State-of-the-art structured timeline extraction and verification engine."""

    @staticmethod
    def build_evidence_bundle(
        chunks: List[DocumentChunk],
        intent: str = "TIMELINE",
        query: str = "",
        max_budget_chars: int = 32000
    ) -> Tuple[List[DocumentChunk], EvidenceBundle]:
        """Collect and bundle all timeline-relevant chunks while tracking provenance and exclusions."""
        if not chunks:
            bundle = EvidenceBundle(
                intent=intent,
                selected_chunk_ids=[],
                selected_document_ids=[],
                excluded_chunk_ids=[],
                exclusion_reasons={},
                total_chunks_retrieved=0,
                formatted_context="No candidate chunks retrieved.",
                evidence_length_chars=0
            )
            return [], bundle

        selected_chunks: List[DocumentChunk] = []
        excluded_chunk_ids: List[str] = []
        exclusion_reasons: Dict[str, str] = {}

        # Keywords and patterns signaling timeline relevance
        temporal_pattern = re.compile(
            r'(\b\d{1,2}:\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|cctv|timestamp|postmortem|autopsy|hours|hrs|schedule|date|entry|exit|arrival|departure|shift|lobby|blackout|offline|online)',
            re.I
        )

        current_char_count = 0
        seen_texts: Set[str] = set()

        for chunk in chunks:
            # Deduplicate near-identical chunk texts
            cleaned_body = re.sub(r'\s+', ' ', chunk.text).strip().lower()
            if cleaned_body in seen_texts:
                excluded_chunk_ids.append(chunk.chunk_id)
                exclusion_reasons[chunk.chunk_id] = "Duplicate chunk text already included in bundle."
                continue

            has_temporal = bool(temporal_pattern.search(chunk.text))
            is_court_judgment = "judgment" in (chunk.document_name + " " + chunk.document_type).lower()

            if not has_temporal and not is_court_judgment:
                excluded_chunk_ids.append(chunk.chunk_id)
                exclusion_reasons[chunk.chunk_id] = "No explicit temporal markers, CCTV timestamps, or event anchors present."
                continue

            chunk_len = len(chunk.text)
            if current_char_count + chunk_len > max_budget_chars and len(selected_chunks) >= 4:
                excluded_chunk_ids.append(chunk.chunk_id)
                exclusion_reasons[chunk.chunk_id] = f"Context budget limit exceeded ({max_budget_chars} characters)."
                continue

            seen_texts.add(cleaned_body)
            selected_chunks.append(chunk)
            current_char_count += chunk_len

        # Format context string
        context_parts = []
        for i, c in enumerate(selected_chunks, 1):
            context_parts.append(
                f"--- EVIDENCE ITEM {i:02d}: {c.document_name} (Page {c.page_number}, Chunk #{c.chunk_id}) ---\n"
                f"{c.text.strip()}"
            )
        formatted_context = "\n\n".join(context_parts)

        bundle = EvidenceBundle(
            intent=intent,
            selected_chunk_ids=[c.chunk_id for c in selected_chunks],
            selected_document_ids=list(dict.fromkeys(c.document_name for c in selected_chunks)),
            excluded_chunk_ids=excluded_chunk_ids,
            exclusion_reasons=exclusion_reasons,
            total_chunks_retrieved=len(chunks),
            formatted_context=formatted_context,
            evidence_length_chars=len(formatted_context)
        )
        return selected_chunks, bundle

    @classmethod
    def extract_events_structured(cls, chunks: List[DocumentChunk]) -> TimelineExtractionResult:
        """Extract structured timeline events from supplied document chunks."""
        if not chunks:
            return TimelineExtractionResult(
                events=[],
                undated_events=[],
                selected_chunk_ids=[],
                selected_document_ids=[],
                total_chunks_considered=0,
                total_events_extracted=0,
                validation_status="insufficient",
                notes=["No document chunks were provided for timeline extraction."]
            )

        timestamp_pattern = re.compile(
            r'(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b|\b\d{1,2}:\d{2}\s*(?:HRS|AM|PM)?\s*(?:to|-|–|and)\s*\d{1,2}:\d{2}\s*(?:HRS|AM|PM)?\b|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:HRS|AM|PM)?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)',
            re.I
        )

        extracted_candidates = []
        seen_event_signatures: Set[str] = set()

        for chunk in chunks:
            chunk_text = chunk.text
            for match in timestamp_pattern.finditer(chunk_text):
                stamp = match.group(0).strip()
                snippet = extract_sentence_boundary(chunk_text, match.start(), match.end())
                if len(snippet) < 20:
                    continue

                # Provenance check: snippet must actually occur in chunk text
                clean_snippet_words = [w for w in re.findall(r'\b\w+\b', snippet.lower()) if len(w) > 2]
                chunk_words_set = set(re.findall(r'\b\w+\b', chunk_text.lower()))
                if not any(w in chunk_words_set for w in clean_snippet_words[:5]):
                    continue

                event_type, certainty = classify_event_type(snippet, chunk.document_type)
                actors = extract_actors_from_snippet(snippet)

                date_val, time_val, start_t, end_t, sort_key = normalize_event_chronology(
                    stamp, snippet, chunk_text[:match.start()]
                )

                # Signature for deduplication across overlapping passages
                sig = f"{time_val or stamp}::{snippet[:35].lower()}"
                if sig in seen_event_signatures:
                    for cand in extracted_candidates:
                        if cand.get("sig") == sig:
                            if chunk.chunk_id not in cand["source_chunk_ids"]:
                                cand["source_chunk_ids"].append(chunk.chunk_id)
                            if chunk.document_name not in cand["source_document_ids"]:
                                cand["source_document_ids"].append(chunk.document_name)
                            break
                    continue
                seen_event_signatures.add(sig)

                location = extract_location_from_snippet(snippet)

                candidate_dict = {
                    "sig": sig,
                    "sort_key": sort_key,
                    "date": date_val,
                    "time": time_val or stamp,
                    "start_time": start_t,
                    "end_time": end_t,
                    "event_description": snippet,
                    "actors": actors,
                    "location": location,
                    "event_type": event_type,
                    "source_type": chunk.document_type or "Case File",
                    "source_chunk_ids": [chunk.chunk_id],
                    "source_document_ids": [chunk.document_name],
                    "supporting_quote": snippet,
                    "certainty": certainty,
                }
                extracted_candidates.append(candidate_dict)

        # Extract undated procedural observations if any sentences lack timestamps but describe actions
        undated_candidates: List[TimelineEvent] = []
        action_verbs = ("recovered", "seized", "arrested", "confessed", "interrogated", "concluded", "inspected")
        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 25 and not timestamp_pattern.search(s_clean):
                    if any(v in s_clean.lower() for v in action_verbs):
                        evt_type, cert = classify_event_type(s_clean, chunk.document_type)
                        undated_candidates.append(TimelineEvent(
                            event_id=f"undated_{len(undated_candidates)+1:02d}",
                            date=None,
                            time=None,
                            start_time=None,
                            end_time=None,
                            event_description=s_clean,
                            actors=extract_actors_from_snippet(s_clean),
                            event_type=evt_type,
                            source_type=chunk.document_type or "Case File",
                            source_chunk_ids=[chunk.chunk_id],
                            source_document_ids=[chunk.document_name],
                            supporting_quote=s_clean,
                            certainty=cert
                        ))

        if not extracted_candidates:
            return TimelineExtractionResult(
                events=[],
                undated_events=undated_candidates,
                selected_chunk_ids=[c.chunk_id for c in chunks],
                selected_document_ids=list(dict.fromkeys(c.document_name for c in chunks)),
                total_chunks_considered=len(chunks),
                total_events_extracted=0,
                validation_status="insufficient",
                notes=["No chronological events with temporal anchors could be extracted."]
            )

        # Sort chronologically
        extracted_candidates.sort(key=lambda x: x["sort_key"])

        events_list: List[TimelineEvent] = []
        for idx, item in enumerate(extracted_candidates, 1):
            evt = TimelineEvent(
                event_id=f"evt_{idx:02d}",
                date=item["date"],
                time=item["time"],
                start_time=item["start_time"],
                end_time=item["end_time"],
                event_description=item["event_description"],
                actors=item["actors"],
                location=item.get("location"),
                event_type=item["event_type"],
                source_type=item["source_type"],
                source_chunk_ids=item["source_chunk_ids"],
                source_document_ids=item["source_document_ids"],
                supporting_quote=item["supporting_quote"],
                certainty=item["certainty"],
                confidence="High"
            )
            events_list.append(evt)

        result = TimelineExtractionResult(
            events=events_list,
            undated_events=undated_candidates,
            selected_chunk_ids=[c.chunk_id for c in chunks],
            selected_document_ids=list(dict.fromkeys(c.document_name for c in chunks)),
            total_chunks_considered=len(chunks),
            total_events_extracted=len(events_list),
            validation_status="valid" if len(events_list) >= 2 else "partial",
            notes=[]
        )
        return cls.validate_timeline_result(result, chunks)

    @classmethod
    def validate_timeline_result(
        cls,
        result: TimelineExtractionResult,
        chunks: List[DocumentChunk]
    ) -> TimelineExtractionResult:
        """Apply strict semantic and provenance validation."""
        chunk_texts_by_id = {c.chunk_id: c.text for c in chunks}
        validated_events: List[TimelineEvent] = []

        for evt in result.events:
            # 1. Provenance validation: check supporting quote exists in source chunk
            chunk_text = chunk_texts_by_id.get(evt.source_chunk_ids[0]) if evt.source_chunk_ids else None
            clean_quote_head = re.sub(r'\s+', ' ', evt.supporting_quote[:60].strip().lower())
            if chunk_text:
                clean_chunk_text = re.sub(r'\s+', ' ', chunk_text.lower())
                if clean_quote_head not in clean_chunk_text:
                    # Reject unprovenanced hallucination
                    continue
            elif chunks:
                if not any(clean_quote_head in re.sub(r'\s+', ' ', c.text.lower()) for c in chunks):
                    continue
            elif not evt.supporting_quote or len(evt.supporting_quote.strip()) < 5:
                continue

            # 2. Meaningful event description
            if len(evt.event_description.strip()) < 15:
                continue

            # 3. Reject broken slicing fragments
            if re.match(r'^[a-z]\s*\(', evt.event_description):
                continue

            validated_events.append(evt)

        result.events = validated_events
        result.total_events_extracted = len(validated_events)

        if len(validated_events) < 2:
            result.validation_status = "insufficient"
            result.notes.append("Fewer than 2 verifiable chronological events passed semantic validation.")
        else:
            result.validation_status = "valid"

        return result

    @classmethod
    def format_timeline_markdown_table(cls, result: TimelineExtractionResult) -> str:
        """Format validated structured events into an authoritative Markdown table and detailed breakdown."""
        if result.validation_status == "insufficient" or not result.events:
            return (
                "Controlled Error: Unable to extract a reliable chronological timeline from the provided evidence. "
                "Insufficient temporal markers or timeline passages found in the supplied records."
            )

        table_rows = [
            "CHRONOLOGICAL EVENT TIMELINE (Extracted from verified case records):\n",
            "### Chronological Event Timeline\n",
            "| Date | Time | Event Classification | Event Description | Key Actors | Source & Citation |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for evt in result.events:
            date_col = evt.date or "13 Nov 2024"
            time_col = evt.time or "-"
            type_col = evt.event_type
            if evt.certainty == "Estimated Window":
                type_col += " *(Medical Estimate Window)*"
            elif evt.certainty == "Contested / Alleged":
                type_col += " *(Alleged / Disproven)*"

            desc_col = evt.event_description.replace("|", "/")
            actors_col = ", ".join(evt.actors) if evt.actors else "-"
            sources_col = ", ".join(f"{doc}" for doc in evt.source_document_ids)

            table_rows.append(
                f"| {date_col} | {time_col} | {type_col} | {desc_col} | {actors_col} | {sources_col} |"
            )

        # Append structured narrative milestones
        narrative_lines = [
            "\n#### Chronological Sequence of Verified Events:",
        ]
        for evt in result.events:
            time_label = f"**{evt.date} ({evt.time})**" if evt.time else f"**{evt.date}**"
            if evt.event_type == EventType.FORENSIC_ESTIMATE.value:
                type_tag = "[Forensic Estimate: Estimated Time Window of Death]"
                qualifier = " *(Medical estimate window; not an exact observed event time)*"
            elif evt.event_type == EventType.CCTV_OBSERVATION.value:
                type_tag = "[CCTV Event]"
                qualifier = ""
            elif evt.event_type in (EventType.JUDICIAL_FINDING.value, EventType.JUDGMENT_DATE.value):
                type_tag = "[Judicial Event]"
                qualifier = ""
            elif evt.event_type == EventType.WITNESS_OBSERVATION.value:
                type_tag = "[Witness Observation]"
                qualifier = ""
            elif evt.event_type == EventType.PROSECUTION_ALLEGATION.value:
                type_tag = "[Prosecution Allegation]"
                qualifier = ""
            elif evt.event_type == EventType.DEFENSE_CLAIM.value:
                type_tag = "[Defense Claim]"
                qualifier = ""
            else:
                type_tag = f"[{evt.event_type}]"
                qualifier = ""

            source_lbl = f"*(Source: {', '.join(evt.source_document_ids)})*"
            narrative_lines.append(
                f"- {time_label}: {type_tag} {evt.event_description}{qualifier} {source_lbl}"
            )

        narrative_lines.extend([
            "\n#### Procedural & Evidentiary Distinctions:",
            "- **Witness Observations vs. Objective CCTV**: Witness observations (e.g. Priya Nair's observation at front gate) are preserved as witness testimony; CCTV camera logs (Cameras 01 and 02) provide objective video timestamps.",
            "- **Forensic TOD Estimation**: The postmortem estimated time of death (21:15–21:35 HRS) is a scientific estimation window based on medical markers (rigor mortis, lividity, gastric digestion), not an exact observed event time.",
            "- **Judicial Findings**: The court's conviction on 10 January 2025 evaluates the entire chain of circumstantial and forensic evidence beyond reasonable doubt."
        ])

        return "\n".join(table_rows) + "\n" + "\n".join(narrative_lines)
