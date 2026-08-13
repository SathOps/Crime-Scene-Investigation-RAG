import re
from typing import List, Dict, Any
from models.schemas import DocumentChunk

class LegalTextSplitter:
    """Legal-aware text chunker that preserves legal section headers, numbered clauses,

    witness statements, and paragraphs while respecting size & overlap constraints.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        for page in pages_data:
            text = page.get("text", "").strip()
            doc_name = page.get("document_name", "Unknown")
            doc_type = page.get("document_type", "Legal Document")
            page_num = page.get("page_number", 1)
            case_name = page.get("case_name", "Unknown Case")

            if not text:
                continue

            # First attempt: split by legal logical boundaries (Headers, double newlines, numbered items)
            sections = self._split_into_logical_sections(text)
            
            # Combine or split sections to fit target chunk_size
            current_chunk_text = ""
            chunk_counter = 1

            for sec in sections:
                if len(current_chunk_text) + len(sec) <= self.chunk_size:
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + sec
                    else:
                        current_chunk_text = sec
                else:
                    if current_chunk_text:
                        chunk_id = f"{doc_name}_p{page_num}_c{chunk_counter}"
                        chunks.append(DocumentChunk(
                            chunk_id=chunk_id,
                            document_name=doc_name,
                            document_type=doc_type,
                            page_number=page_num,
                            case_name=case_name,
                            text=current_chunk_text.strip()
                        ))
                        chunk_counter += 1
                        
                        # Keep overlap text
                        overlap_start = max(0, len(current_chunk_text) - self.chunk_overlap)
                        current_chunk_text = current_chunk_text[overlap_start:] + "\n\n" + sec
                    else:
                        # Section itself exceeds chunk_size, split by sentences or hard break
                        sub_chunks = self._hard_split_text(sec)
                        for sub in sub_chunks:
                            chunk_id = f"{doc_name}_p{page_num}_c{chunk_counter}"
                            chunks.append(DocumentChunk(
                                chunk_id=chunk_id,
                                document_name=doc_name,
                                document_type=doc_type,
                                page_number=page_num,
                                case_name=case_name,
                                text=sub.strip()
                            ))
                            chunk_counter += 1
                        current_chunk_text = ""

            if current_chunk_text.strip():
                chunk_id = f"{doc_name}_p{page_num}_c{chunk_counter}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=doc_name,
                    document_type=doc_type,
                    page_number=page_num,
                    case_name=case_name,
                    text=current_chunk_text.strip()
                ))

        return chunks

    def _split_into_logical_sections(self, text: str) -> List[str]:
        # Regex patterns for legal sections, witness titles, numbered lists
        pattern = r'(\n(?=[A-Z0-9\.\-\s]{3,40}:)|\n(?=SECTION\s+\d+)|\n(?=WITNESS\s+\d+)|\n(?=\d+\.\s+[A-Z])|\n\n+)'
        parts = re.split(pattern, text)
        sections = []
        for p in parts:
            p_clean = p.strip()
            if p_clean:
                sections.append(p_clean)
        return sections if sections else [text]

    def _hard_split_text(self, text: str) -> List[str]:
        # Fallback split by sentences or fixed character window
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        curr = ""
        for s in sentences:
            if len(curr) + len(s) <= self.chunk_size:
                curr += (" " + s) if curr else s
            else:
                if curr:
                    chunks.append(curr)
                curr = s
        if curr:
            chunks.append(curr)
        return chunks
