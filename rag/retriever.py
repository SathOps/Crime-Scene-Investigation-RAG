import re
from typing import List, Dict, Any, Tuple, Optional, Set
from models.schemas import DocumentChunk
from rag.vector_store import VectorStoreManager
from rag.embeddings import EmbeddingService

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", 
    "of", "and", "or", "it", "this", "that", "what", "which", "who", "when", 
    "where", "how", "why", "did", "does", "do", "have", "has", "had", "be", 
    "been", "being", "with", "as", "by", "from", "at", "about", "into", "through"
}

class RAGRetriever:
    """Enhanced Retriever executing hybrid semantic + lexical search,
    entity-preserving boosting, multi-query expansion, and chunk deduplication.
    """

    def __init__(self, vector_store: VectorStoreManager, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 5,
        normalized_query: Optional[str] = None,
        entities: Optional[List[str]] = None,
        intent: Optional[str] = None
    ) -> Tuple[List[DocumentChunk], str]:
        """Retrieves evidence with hybrid scoring and entity preservation."""
        search_query = (normalized_query or query).strip()
        candidates: Dict[str, DocumentChunk] = {}

        # 1. Primary Vector Search (Fetch larger pool for hybrid reranking)
        pool_size = max(top_k * 3, 10)
        primary_chunks = self.vector_store.search(search_query, top_k=pool_size, embedding_service=self.embedding_service)
        for c in primary_chunks:
            candidates[c.chunk_id] = c

        # 2. Multi-query Expansion for Compound Queries (e.g., Witness vs CCTV)
        sub_queries = self._generate_sub_queries(search_query, intent)
        for sq in sub_queries:
            sq_chunks = self.vector_store.search(sq, top_k=max(top_k, 5), embedding_service=self.embedding_service)
            for sc in sq_chunks:
                if sc.chunk_id not in candidates:
                    candidates[sc.chunk_id] = sc

        if not candidates:
            return [], "Insufficient"

        # 3. Lexical & Entity Scoring
        query_terms = self._extract_query_keywords(search_query)
        target_entities = [e.lower() for e in (entities or [])]
        max_vec_score = max((c.score for c in candidates.values()), default=0.0)

        scored_chunks: List[Tuple[float, DocumentChunk]] = []
        for chunk in candidates.values():
            vec_score = chunk.score
            chunk_lower = chunk.text.lower()

            # Lexical Term Frequency Match
            term_matches = sum(1 for term in query_terms if term in chunk_lower)
            lexical_ratio = (term_matches / len(query_terms)) if query_terms else 0.0

            # Exact Entity Boost
            entity_boost = 0.0
            for ent in target_entities:
                if ent in chunk_lower:
                    entity_boost += 0.20

            # Topic-aware boosts prefer substantive result/order passages over headers.
            q_lower = search_query.lower()
            doc_lower = (chunk.document_name + " " + chunk.document_type).lower()
            if "cctv" in q_lower and "cctv" in doc_lower:
                entity_boost += 0.15
            if any(term in q_lower for term in ("witness", "guard", "contradict", "discrepan", "inconsisten", "statement", "testimony")):
                if "witness" in doc_lower or "statement" in doc_lower:
                    entity_boost += 0.18
                if any(w in chunk_lower for w in ("witness", "statement", "deposed", "testified", "contradiction", "discrepancy")):
                    entity_boost += 0.10
                if "judgment" in doc_lower and any(w in chunk_lower for w in ("contradiction", "witness statements", "honest error")):
                    entity_boost += 0.20
            if any(term in q_lower for term in ("judgment", "judgement", "verdict", "court finding", "final order")):
                if "judgment" in doc_lower or "court" in doc_lower:
                    entity_boost += 0.22
                    if any(marker in chunk_lower for marker in ("final order", "the court finds", "the court held", "ordered that", "acquitted", "convicted", "sentence", "reasons for decision", "findings of the court", "operative order")):
                        entity_boost += 0.28
                    if "prosecution case summary" in chunk_lower:
                        entity_boost -= 0.12
            if any(term in q_lower for term in ("forensic", "laboratory findings", "ballistic", "fingerprint", "dna", "g.s.r", "g sr")):
                if any(term in doc_lower for term in ("forensic", "laboratory", "medical")):
                    entity_boost += 0.22
                    if any(marker in chunk_lower for marker in ("conclusion", "opinion", "matched", "excluded", "identified", "consistent with", "inconclusive", "detected", "not detected", "analysis revealed", "results:")):
                        entity_boost += 0.28
                    if chunk_lower.rstrip().endswith("laboratory findings:"):
                        entity_boost -= 0.10
            
            # Timestamp matching boost (e.g. 21:25, 21:10, 21:40)
            timestamps_in_query = re.findall(r'\b\d{1,2}:\d{2}\b', search_query.lower())
            for ts in timestamps_in_query:
                if ts in chunk_lower:
                    entity_boost += 0.30

            if any(term in search_query.lower() for term in ("timeline", "chronology", "sequence", "events", "when did", "time of")) and re.search(r"\b\d{1,2}:\d{2}\b", chunk_lower):
                entity_boost += 0.15

            # Hybrid Score calculation: 55% semantic + 45% lexical/entity
            lexical_score = min(1.0, lexical_ratio + entity_boost)
            hybrid_score = round(min(1.0, 0.55 * vec_score + 0.45 * lexical_score), 3)

            # Create a clone so original vector-store chunks are not mutated in place
            scored_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                document_type=chunk.document_type,
                page_number=chunk.page_number,
                text=chunk.text,
                score=hybrid_score
            )
            scored_chunks.append((hybrid_score, scored_chunk))

        # 4. Deduplication & Multi-Source Diversity Reranking
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        deduped_chunks: List[DocumentChunk] = []
        seen_texts: Set[str] = set()
        doc_counts: Dict[str, int] = {}
        max_chunks_per_doc = max(2, top_k // 2) if top_k >= 4 else top_k

        # Pass 1: Select top chunks while enforcing document diversity cap
        remaining_candidates: List[DocumentChunk] = []
        for score, chunk in scored_chunks:
            snippet = chunk.text[:120].strip()
            if snippet in seen_texts:
                continue
            seen_texts.add(snippet)

            doc_name = chunk.document_name or "Unknown"
            if doc_counts.get(doc_name, 0) < max_chunks_per_doc:
                deduped_chunks.append(chunk)
                doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
                if len(deduped_chunks) >= top_k:
                    break
            else:
                remaining_candidates.append(chunk)

        # Pass 2: If diversity cap left empty slots, fill from remaining highest scoring candidates
        if len(deduped_chunks) < top_k:
            for chunk in remaining_candidates:
                deduped_chunks.append(chunk)
                if len(deduped_chunks) >= top_k:
                    break

        # 5. Calibrate Evidence Strength Thresholds
        if not deduped_chunks:
            return [], "Insufficient"

        avg_score = sum(c.score for c in deduped_chunks) / len(deduped_chunks)
        max_score = max(c.score for c in deduped_chunks)

        # Check for meaningful keyword coverage across retrieved chunks
        best_term_matches = max(
            sum(1 for term in query_terms if term in c.text.lower())
            for c in deduped_chunks
        ) if query_terms else 0
        best_term_ratio = (best_term_matches / len(query_terms)) if query_terms else 0.0
        has_entity_match = any(any(ent in c.text.lower() for ent in target_entities) for c in deduped_chunks) if target_entities else False

        # Intent & Content checks for generalized legal reasoning
        q_lower = search_query.lower()
        is_timeline_query = any(term in q_lower for term in ("timeline", "chronolog", "sequence", "events", "when did", "time of")) or (intent and "TIMELINE" in intent.upper())
        timestamps_count = sum(len(re.findall(r'\b\d{1,2}:\d{2}\b', c.text)) for c in deduped_chunks)
        dates_count = sum(len(re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', c.text, re.I)) for c in deduped_chunks)
        has_temporal_density = (timestamps_count + dates_count) >= 3

        is_contradiction_query = any(term in q_lower for term in ("contradict", "discrepan", "inconsisten", "conflict")) or (intent and "CONTRADICTION" in intent.upper())
        has_witness_context = any(
            any(k in c.text.lower() for k in ("witness", "testimony", "statement", "observation", "deposed", "contradiction", "discrepancy"))
            for c in deduped_chunks
        )

        # Insufficient threshold: off-topic or absent record questions have poor overlap and no entity match
        if is_timeline_query and has_temporal_density:
            # Timeline query with substantive temporal markers must not be rejected on keyword mismatch alone
            if max_score >= 0.50 or avg_score >= 0.40 or timestamps_count >= 3:
                strength = "High" if (max_score >= 0.60 or timestamps_count >= 4) else "Medium"
            else:
                strength = "Low"
        elif is_contradiction_query and has_witness_context and max_score >= 0.38:
            # Contradiction query with witness context must not be rejected on missing lexical tokens
            strength = "High" if (max_score >= 0.60 or avg_score >= 0.45) else "Medium"
        elif max_score < 0.40 or (best_term_ratio < 0.25 and not has_entity_match and max_score < 0.52 and max_vec_score < 0.65):
            strength = "Insufficient"
        elif max_score >= 0.65 or avg_score >= 0.50:
            strength = "High"
        elif max_score >= 0.46 or avg_score >= 0.38:
            strength = "Medium"
        elif max_score >= 0.40:
            strength = "Low"
        else:
            strength = "Insufficient"

        # Similarity alone is not enough for a conclusion-specific legal question.
        if any(term in q_lower for term in ("judgment", "judgement", "verdict", "court finding", "final order")):
            court_text = " ".join(c.text.lower() for c in deduped_chunks
                                  if "judgment" in (c.document_name + " " + c.document_type).lower()
                                  or "court" in (c.document_name + " " + c.document_type).lower())
            if not re.search(r"\b(acquitted|convicted|found guilty|not guilty|sentenced to|operative order|the court held|the court finds)\b", court_text):
                strength = "Low" if strength != "Insufficient" else strength
        if any(term in q_lower for term in ("forensic", "laboratory findings", "ballistic", "fingerprint", "dna", "g.s.r", "g sr")):
            forensic_text = " ".join(c.text.lower() for c in deduped_chunks
                                     if any(t in (c.document_name + " " + c.document_type).lower()
                                            for t in ("forensic", "laboratory", "medical")))
            if not re.search(r"\b(conclusion|opinion|matched|excluded|identified|consistent with|inconclusive|detected|not detected|positive|negative|analysis revealed)\b", forensic_text):
                strength = "Low" if strength != "Insufficient" else strength

        return deduped_chunks, strength

    @staticmethod
    def get_unique_document_count(chunks: List[DocumentChunk]) -> int:
        """Count distinct source document files represented in chunk list."""
        if not chunks:
            return 0
        return len({c.document_name for c in chunks if c.document_name})

    def _extract_query_keywords(self, query: str) -> List[str]:
        q_lower = query.lower()
        words = re.findall(r'\b[a-zA-Z0-9_\-\#]+\b', q_lower)
        terms = [w for w in words if w not in STOP_WORDS and (len(w) > 2 or w.isdigit())]
        
        # Include explicit timestamps like 21:25, 21:40
        timestamps = re.findall(r'\b\d{1,2}:\d{2}\b', q_lower)
        terms.extend(timestamps)
        return list(set(terms))

    def _generate_sub_queries(self, query: str, intent: Optional[str]) -> List[str]:
        """Generate case-agnostic retrieval prompts without injecting sample-case facts."""
        q_lower = (query or "").lower()
        sub_queries: List[str] = []

        if any(term in q_lower for term in ("what happened", "case summary", "facts of the case", "overview", "prosecution", "connects the accused")):
            sub_queries.extend([
                "prosecution case summary allegations incident facts charges evidence",
                "brief facts of the case incident FIR complaint investigation findings",
            ])

        if any(term in q_lower for term in ("contradict", "discrepan", "inconsisten", "conflict", "witness", "statement")):
            sub_queries.extend([
                "witness statement testimony observations date time clothing entrance",
                "witness statement contradiction discrepancy CCTV camera corroboration",
                "evaluation of witness statements and contradictions court finding",
            ])

        if any(term in q_lower for term in ("timeline", "chronolog", "sequence", "events", "when did", "time of")):
            sub_queries.extend([
                "chronological sequence of events dates timestamps incident murder entry exit",
                "timeline of events hours timestamps CCTV witness postmortem",
            ])

        if "alibi" in q_lower:
            sub_queries.extend([
                "accused alibi statement location date time",
                "call detail records cell tower location time",
            ])

        if any(term in q_lower for term in ("forensic", "laboratory findings", "ballistic", "fingerprint", "dna", "g.s.r", "g sr")):
            sub_queries.extend([
                "forensic laboratory findings results conclusion opinion",
                "forensic analysis matched excluded detected not detected",
            ])

        if any(term in q_lower for term in ("judgment", "judgement", "verdict", "court finding", "final order", "acquitted", "convicted")):
            sub_queries.extend([
                "court judgment final order judicial findings reasons verdict sentence",
                "operative order convicted acquitted sentenced court held",
            ])

        if any(term in q_lower for term in ("weapon", "gun", "casing", "ballistic")):
            sub_queries.append("forensic laboratory weapon firearm ballistic examination results")

        return list(dict.fromkeys(sub_queries))

    def format_chunks_for_context(self, chunks: List[DocumentChunk]) -> str:
        if not chunks:
            return "NO RELEVANT DOCUMENTS FOUND."

        formatted_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"--- EVIDENCE ITEM [{idx}] ---\n"
                f"Source File: {chunk.document_name}\n"
                f"Document Type: {chunk.document_type}\n"
                f"Page Number: {chunk.page_number}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Relevance Score: {chunk.score}\n"
                f"Content:\n{chunk.text}\n"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)
