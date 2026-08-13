import os
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import docx

class DocumentLoader:
    """Loads PDF, DOCX, and TXT legal documents with metadata extraction."""

    @staticmethod
    def detect_document_type(filename: str, content: str) -> str:
        filename_lower = filename.lower()
        content_lower = content[:1000].lower()

        if "fir" in filename_lower or "first information report" in content_lower:
            return "FIR"
        elif "charge" in filename_lower or "chargesheet" in filename_lower or "charge sheet" in content_lower:
            return "Charge Sheet"
        elif "witness" in filename_lower or "statement" in filename_lower or "witness statement" in content_lower:
            return "Witness Statement"
        elif "judgment" in filename_lower or "order" in filename_lower or "court" in content_lower:
            return "Court Judgment / Order"
        elif "forensic" in filename_lower or "autopsy" in filename_lower or "medical" in content_lower:
            return "Forensic / Medical Report"
        elif "statute" in filename_lower or "act" in filename_lower or "section" in content_lower:
            return "Statute / Legal Act"
        else:
            return "Legal Document"

    @classmethod
    def load_pdf(cls, file_path_or_stream, filename: str) -> List[Dict[str, Any]]:
        pages_data = []
        try:
            reader = pypdf.PdfReader(file_path_or_stream)
            full_text = ""
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                full_text += text + "\n"
                pages_data.append({
                    "text": text.strip(),
                    "page_number": idx + 1,
                    "document_name": filename,
                })
            
            doc_type = cls.detect_document_type(filename, full_text)
            for page in pages_data:
                page["document_type"] = doc_type
                page["case_name"] = Path(filename).stem
        except Exception as e:
            raise ValueError(f"Failed to read PDF file '{filename}': {str(e)}")
        return pages_data

    @classmethod
    def load_docx(cls, file_path_or_stream, filename: str) -> List[Dict[str, Any]]:
        try:
            doc = docx.Document(file_path_or_stream)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            doc_type = cls.detect_document_type(filename, full_text)
            
            # Group into synthetic pages (approx 500 words per page)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            pages_data = []
            current_page_text = []
            word_count = 0
            page_num = 1

            for p in paragraphs:
                current_page_text.append(p)
                word_count += len(p.split())
                if word_count >= 400:
                    pages_data.append({
                        "text": "\n".join(current_page_text),
                        "page_number": page_num,
                        "document_name": filename,
                        "document_type": doc_type,
                        "case_name": Path(filename).stem
                    })
                    page_num += 1
                    current_page_text = []
                    word_count = 0

            if current_page_text:
                pages_data.append({
                    "text": "\n".join(current_page_text),
                    "page_number": page_num,
                    "document_name": filename,
                    "document_type": doc_type,
                    "case_name": Path(filename).stem
                })
        except Exception as e:
            raise ValueError(f"Failed to read DOCX file '{filename}': {str(e)}")
        return pages_data

    @classmethod
    def load_txt(cls, file_path_or_stream, filename: str) -> List[Dict[str, Any]]:
        try:
            if isinstance(file_path_or_stream, (str, Path)):
                with open(file_path_or_stream, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                content = file_path_or_stream.read().decode("utf-8", errors="ignore")

            doc_type = cls.detect_document_type(filename, content)
            
            # Split into synthetic pages by double newlines or line count
            lines = content.splitlines()
            pages_data = []
            page_num = 1
            lines_per_page = 45

            for i in range(0, len(lines), lines_per_page):
                page_lines = lines[i:i + lines_per_page]
                page_text = "\n".join(page_lines).strip()
                if page_text:
                    pages_data.append({
                        "text": page_text,
                        "page_number": page_num,
                        "document_name": filename,
                        "document_type": doc_type,
                        "case_name": Path(filename).stem
                    })
                    page_num += 1
        except Exception as e:
            raise ValueError(f"Failed to read TXT file '{filename}': {str(e)}")
        return pages_data

    @classmethod
    def load_file(cls, file_path_or_stream, filename: str) -> List[Dict[str, Any]]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path_or_stream, filename)
        elif ext in [".docx", ".doc"]:
            return cls.load_docx(file_path_or_stream, filename)
        elif ext in [".txt", ".md", ".log"]:
            return cls.load_txt(file_path_or_stream, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Allowed formats: PDF, DOCX, TXT.")
