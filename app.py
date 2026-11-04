import sys
import os

# Linux / Streamlit Community Cloud modern SQLite shim for ChromaDB
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

# Suppress tqdm / transformers progress bars that cause Errno 5 on detached stderr
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _stream in (sys.stdout, sys.stderr):
    _orig = getattr(_stream, "flush", None)
    if _orig:
        def _safe_flush(_f=_orig):
            try:
                _f()
            except OSError:
                pass
        _stream.flush = _safe_flush

try:
    import transformers
    transformers.utils.logging.disable_progress_bar()
except Exception:
    pass

import html
import re
from pathlib import Path
import streamlit as st
import json
import time
from datetime import datetime

from config import (
    DOCUMENTS_DIR, VECTORSTORE_DIR, SAMPLE_DATA_DIR,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K,
    DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, DEFAULT_GEMINI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_API_KEY, OPENAI_API_KEY
)
import importlib
import models.schemas
importlib.reload(models.schemas)
from models.schemas import WorkflowState, LegalResearchAnswer
from rag.document_loader import DocumentLoader
from rag.text_splitter import LegalTextSplitter
import rag.embeddings
importlib.reload(rag.embeddings)
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
import rag.retriever
importlib.reload(rag.retriever)
from rag.retriever import RAGRetriever
from rag.evaluation import evaluate_retrieval, benchmark_retrieval, DEFAULT_LEGAL_EVALUATION_CASES
from rag.benchmark import compare_llm_only_vs_rag
import llm.llm_service
importlib.reload(llm.llm_service)
from llm.llm_service import LLMService
import agent.timeline_pipeline
importlib.reload(agent.timeline_pipeline)
import agent.contradiction_pipeline
importlib.reload(agent.contradiction_pipeline)
import agent.tools
importlib.reload(agent.tools)
import agent.specialist_agents
importlib.reload(agent.specialist_agents)
import agent.legal_research_agent
importlib.reload(agent.legal_research_agent)
from agent.legal_research_agent import LegalResearchAgent, is_standalone_hypothetical_query
from utils.helpers import state_to_json, format_file_size

def format_readable_answer(raw_answer: str) -> str:
    """Enhance raw legal answers with colorful badges, quotes, timestamps, citations, and structured sections."""
    if not raw_answer:
        return ""
    
    text = raw_answer

    # 1. Blockquote quoted witness/accused speech before injecting HTML attributes
    quote_pattern = re.compile(r"([\"“]([A-Z][^\"”\n]{15,})[\"”])")
    text = quote_pattern.sub(r'<blockquote class="evidence-quote">“\2”</blockquote>', text)

    # 2a. Convert "According to <doc> (Page <p>):" into a high-visibility source banner
    doc_banner_pattern = re.compile(
        r"(?:^|\n\n?)(?:According to|From|Per)\s+([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*\((?:Page|p\.|page)?\s*(\d+)[^)]*\):?",
        re.IGNORECASE
    )
    def _doc_banner_sub(m):
        return f'\n\n<div class="evidence-source-card-header"><span class="evidence-source-icon">📄</span><span class="evidence-source-doc">{m.group(1)}</span><span class="evidence-source-page">PAGE {m.group(2)}</span></div>\n\n'
    text = doc_banner_pattern.sub(_doc_banner_sub, text)

    # 2b. Standalone or line-start parenthesized citations: "(03_CCTV_Analysis_Report.txt, Page 1)"
    line_start_citation_pattern = re.compile(
        r"(?:^|\n\n?)\s*(?:[•\-*]|\d+\.)?\s*[\(\[]\s*([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*(?:,\s*(?:Page|p\.|page)?\s*(\d+))?(?:\s*;\s*chunk\s*\d+)?\s*[\)\]]:?",
        re.IGNORECASE
    )
    def _line_start_sub(m):
        doc = m.group(1)
        page = f"PAGE {m.group(2)}" if m.group(2) else "DOC"
        return f'\n\n<div class="evidence-source-card-header"><span class="evidence-source-icon">📄</span><span class="evidence-source-doc">{doc}</span><span class="evidence-source-page">{page}</span></div>\n\n'
    text = line_start_citation_pattern.sub(_line_start_sub, text)

    # 2c. Evidence Passage references: "Evidence Passage 01 (doc, Page 1):"
    passage_pat = re.compile(
        r"\bEvidence\s+Passage\s+(\d+)\s*[\(\[]\s*([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*(?:,\s*(?:Page|p\.|page)?\s*(\d+))?(?:\s*;\s*chunk\s*\d+)?\s*[\)\]]:?",
        re.IGNORECASE
    )
    def _passage_sub(m):
        num = m.group(1)
        doc = m.group(2)
        page_badge = f'<span class="citation-divider">|</span><span class="citation-page">PAGE {m.group(3)}</span>' if m.group(3) else ""
        return f'<span class="passage-badge">PASSAGE {num}</span><span class="citation-pill"><span class="citation-icon">📄</span><span class="citation-doc">{doc}</span>{page_badge}</span> '
    text = passage_pat.sub(_passage_sub, text)

    # 2d. Remaining inline citations: "(03_CCTV_Analysis_Report.txt, Page 1)" or "[doc, p. 1]"
    inline_citation_pat = re.compile(
        r"[\(\[]\s*([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*(?:,\s*(?:Page|p\.|page)?\s*(\d+))?(?:\s*;\s*chunk\s*\d+)?\s*[\)\]]",
        re.IGNORECASE
    )
    def _inline_sub(m):
        doc = m.group(1)
        page_badge = f'<span class="citation-divider">|</span><span class="citation-page">PAGE {m.group(2)}</span>' if m.group(2) else ""
        return f'<span class="citation-pill"><span class="citation-icon">📄</span><span class="citation-doc">{doc}</span>{page_badge}</span>'
    text = inline_citation_pat.sub(_inline_sub, text)

    # 3. Timestamps
    timestamp_pattern = re.compile(
        r"(?:•\s*)?Timestamp\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:HRS|hrs|AM|PM|am|pm)?)\s*(?:\(([^)]+)\))?:?",
        re.IGNORECASE
    )
    def _ts_sub(m):
        date_str = f' <span class="timestamp-date">({m.group(2)})</span>' if m.group(2) else ""
        return f'<div class="timeline-ts-row"><span class="timestamp-badge">🕒 {m.group(1)}</span>{date_str}</div>\n'
    text = timestamp_pattern.sub(_ts_sub, text)

    # 4. Q&A highlights
    text = re.sub(r"\b(Q\d+|Question\s*\d*)\s*:\s*", r'<span class="qa-badge-q">\1:</span> ', text)
    text = re.sub(r"\bAnswer\s*:\s*", r'<span class="qa-badge-a">Answer:</span> ', text)

    # 5. Statutes & Acts
    statute_pattern = re.compile(
        r"\b((?:Section|Sec\.|Sections)\s+\d+[A-Z]?(?:\s+(?:Cr\.?P\.?C\.?|BNSS|IPC|IEA|BSA))?(?:[,\s]+and\s+\d+[A-Z]?)*)\b",
        re.IGNORECASE
    )
    text = statute_pattern.sub(r'<span class="statute-pill">\1</span>', text)

    return text


def format_finding_item(text: str) -> str:
    """Enhance findings and evidence summaries with colorful citation badges and highlights."""
    if not text:
        return ""
    formatted = text

    # 1. Evidence Passage XX
    passage_pat = re.compile(
        r"\bEvidence\s+Passage\s+(\d+)\s*[\(\[]\s*([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*(?:,\s*(?:Page|p\.|page)?\s*(\d+))?(?:\s*;\s*chunk\s*\d+)?\s*[\)\]]:?",
        re.IGNORECASE
    )
    def _passage_sub(m):
        num = m.group(1)
        doc = m.group(2)
        page_badge = f'<span class="citation-divider">|</span><span class="citation-page">PAGE {m.group(3)}</span>' if m.group(3) else ""
        return f'<span class="passage-badge">PASSAGE {num}</span><span class="citation-pill"><span class="citation-icon">📄</span><span class="citation-doc">{doc}</span>{page_badge}</span> '
    formatted = passage_pat.sub(_passage_sub, formatted)

    # 2. General citations: (03_CCTV_Analysis_Report.txt, Page 1) or [07_Accused_Statement.txt, p. 1]
    citation_pat = re.compile(
        r"[\(\[]\s*([a-zA-Z0-9_\-\.]+\.(?:txt|pdf|docx|csv|json|png|jpg|jpeg)|\d{2}_[a-zA-Z0-9_\-]+)\s*(?:,\s*(?:Page|p\.|page)?\s*(\d+))?(?:\s*;\s*chunk\s*\d+)?\s*[\)\]]:?",
        re.IGNORECASE
    )
    def _cite_sub(m):
        doc = m.group(1)
        page_badge = f'<span class="citation-divider">|</span><span class="citation-page">PAGE {m.group(2)}</span>' if m.group(2) else ""
        return f'<span class="citation-pill"><span class="citation-icon">📄</span><span class="citation-doc">{doc}</span>{page_badge}</span> '
    formatted = citation_pat.sub(_cite_sub, formatted)

    # 3. Statutes
    statute_pattern = re.compile(
        r"\b((?:Section|Sec\.|Sections)\s+\d+[A-Z]?(?:\s+(?:Cr\.?P\.?C\.?|BNSS|IPC|IEA|BSA))?(?:[,\s]+and\s+\d+[A-Z]?)*)\b",
        re.IGNORECASE
    )
    formatted = statute_pattern.sub(r'<span class="statute-pill">\1</span>', formatted)
    formatted = re.sub(r"^(Contradiction Identified\s*:)", r'<strong style="color:#EF4444;">⚠️ \1</strong>', formatted, flags=re.IGNORECASE)
    formatted = re.sub(r"^(Verified Procedural Record\s*:)", r'<strong style="color:#60A5FA;">✓ \1</strong>', formatted, flags=re.IGNORECASE)
    return formatted

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CSI RAG — Legal Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM DESIGN SYSTEM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

/* ── DESIGN TOKENS ────────────────────────── */
:root {
    --bg-deep:      #05070A;
    --bg-base:      #070A0F;
    --bg-elevated:  #0A0D12;
    --surface-1:    #0D1118;
    --surface-2:    #111620;
    --surface-3:    #151A23;
    --border-subtle:rgba(255,255,255,0.06);
    --border-mid:   rgba(255,255,255,0.10);
    --border-blue:  rgba(37,99,235,0.25);
    --border-blue-h:rgba(59,130,246,0.45);
    --blue-primary: #2563EB;
    --blue-mid:     #2979FF;
    --blue-bright:  #3B82F6;
    --blue-glow:    #60A5FA;
    --blue-dim:     rgba(37,99,235,0.08);
    --blue-dim-2:   rgba(37,99,235,0.15);
    --green:        #22C55E;
    --amber:        #F59E0B;
    --text-primary: #F5F7FA;
    --text-secondary:#9CA3AF;
    --text-muted:   #5E6878;
    --font-sans:    'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono:    'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    14px;
    --transition:   all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── GLOBAL RESET ─────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"],
[data-testid="stMainBlockContainer"] {
    background-color: var(--bg-deep) !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
}
[data-testid="stHeader"] {
    background: var(--bg-deep) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}
.main .block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1440px;
}

/* ── SCROLLBAR ────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue-primary); }

/* ── SIDEBAR ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-base) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
}

/* ── BUTTONS ──────────────────────────────── */
.stButton > button {
    background: var(--surface-2) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    padding: 8px 16px !important;
    min-height: 38px !important;
    text-transform: uppercase !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
.stButton > button:hover {
    border-color: var(--blue-primary) !important;
    color: var(--blue-bright) !important;
    background: var(--blue-dim) !important;
    box-shadow: 0 0 12px rgba(37,99,235,0.12) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary Action Button (e.g. Process in sidebar) - matched height */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue-primary) 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid var(--blue-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 8px 16px !important;
    min-height: 38px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.2) !important;
    box-sizing: border-box !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--blue-bright) 0%, var(--blue-primary) 100%) !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Main Form Submit Button (Run Analysis) */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, var(--blue-primary) 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid var(--blue-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    min-height: 44px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.2) !important;
    box-sizing: border-box !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, var(--blue-bright) 0%, var(--blue-primary) 100%) !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── INPUTS ───────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    padding: 12px 14px !important;
    transition: var(--transition) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--blue-primary) !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.15), 0 0 16px rgba(37,99,235,0.08) !important;
    outline: none !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── SLIDER ───────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--blue-primary) !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
    background: var(--blue-primary) !important;
}

/* ── FILE UPLOADER ────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface-1) !important;
    border: 1px dashed var(--border-blue) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue-bright) !important;
    background: var(--blue-dim) !important;
}
[data-testid="stFileUploader"] button {
    text-transform: none !important;
    font-family: var(--font-sans) !important;
    letter-spacing: normal !important;
}
[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
}

/* ── TABS ─────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
    transition: var(--transition) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--blue-bright) !important;
    border-bottom: 2px solid var(--blue-primary) !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] {
    padding-top: 1.5rem !important;
}

/* ── EXPANDERS ─────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 12px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    padding: 14px 18px !important;
    background: var(--surface-2) !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--blue-bright) !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 16px 18px !important;
}

/* ── POPOVER ──────────────────────────────── */
[data-testid="stPopover"] > button {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 14px !important;
}
[data-testid="stPopoverBody"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
}

/* ── ALERTS ───────────────────────────────── */
[data-testid="stAlert"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
}

/* ── SELECT BOX DROPDOWN ──────────────────── */
[data-baseweb="popover"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-mid) !important;
}
[data-baseweb="menu"] {
    background: var(--surface-1) !important;
}
[data-baseweb="menu"] li {
    background: var(--surface-1) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="menu"] li:hover {
    background: var(--blue-dim-2) !important;
}

/* ═══════════════════════════════════════════ */
/*  CUSTOM COMPONENT CLASSES                  */
/* ═══════════════════════════════════════════ */

/* ── ANIMATIONS ───────────────────────────── */
@keyframes pulse-online {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px var(--green); }
    50%      { opacity: 0.5; box-shadow: 0 0 8px var(--green), 0 0 16px rgba(34,197,94,0.3); }
}
@keyframes pulse-blue {
    0%, 100% { opacity: 1; box-shadow: 0 0 4px var(--blue-primary); }
    50%      { opacity: 0.6; box-shadow: 0 0 10px var(--blue-primary), 0 0 20px rgba(37,99,235,0.2); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
@keyframes grid-drift {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* ── TOP BAR ──────────────────────────────── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 24px;
    animation: fadeInUp 0.5s ease;
}
.top-bar-left {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.top-bar-left strong {
    color: var(--blue-bright);
    font-weight: 600;
}
.top-bar-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.top-bar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.online {
    background: var(--green);
    animation: pulse-online 2.5s ease-in-out infinite;
}
.status-dot.ready {
    background: var(--blue-bright);
    animation: pulse-blue 3s ease-in-out infinite;
}

/* ── HERO ─────────────────────────────────── */
.hero-section {
    position: relative;
    padding: 40px 0 32px;
    margin-bottom: 8px;
    animation: fadeInUp 0.6s ease;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 500px; height: 500px;
    transform: translate(-50%, -50%);
    background: radial-gradient(circle, rgba(37,99,235,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.hero-eyebrow {
    position: relative; z-index: 1;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--blue-primary);
    margin-bottom: 12px;
}
.hero-title {
    position: relative; z-index: 1;
    font-family: var(--font-sans);
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    color: var(--text-primary);
    margin-bottom: 8px;
    white-space: nowrap;
}
.hero-title .highlight {
    color: var(--blue-bright);
    text-shadow: 0 0 40px rgba(59,130,246,0.2);
}
.hero-sub {
    position: relative; z-index: 1;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

/* ── SYSTEM NOTICE ────────────────────────── */
.system-notice {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--blue-primary);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    margin: 16px 0 24px;
    animation: fadeInUp 0.7s ease;
}
.notice-indicator {
    width: 8px; height: 8px; min-width: 8px;
    border-radius: 50%;
    background: var(--blue-primary);
    margin-top: 4px;
    animation: pulse-blue 3s ease-in-out infinite;
}
.notice-content {
    flex: 1;
}
.notice-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue-bright);
    margin-bottom: 4px;
}
.notice-text {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── SECTION LABELS ───────────────────────── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding-bottom: 8px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-subtle);
}
.section-label-blue {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--blue-primary);
    padding-bottom: 8px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border-blue);
}

/* ── METRIC CARDS (inline) ────────────────── */
.metrics-bar {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    animation: fadeInUp 0.6s ease 0.1s both;
}
.metric-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    flex: 1;
    min-width: 0;
    transition: var(--transition);
}
.metric-card:hover {
    border-color: var(--border-blue);
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── PROMPT CARDS ─────────────────────────── */
.prompt-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    transition: var(--transition);
    height: 74px;
    box-sizing: border-box;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.prompt-card:hover {
    border-color: var(--blue-primary);
    background: var(--blue-dim);
}
.prompt-num {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--blue-primary);
    letter-spacing: 0.1em;
    margin-bottom: 3px;
    line-height: 1;
}
.prompt-title {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 3px;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prompt-desc {
    font-family: var(--font-sans);
    font-size: 0.62rem;
    color: var(--text-muted);
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── COMMAND INPUT ────────────────────────── */
.command-meta {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 0;
    margin-top: 4px;
}
.command-tag {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 3px;
    padding: 3px 8px;
}
.command-tag.active {
    color: var(--green);
    border-color: rgba(34,197,94,0.2);
    background: rgba(34,197,94,0.06);
}

/* ── WORKFLOW PIPELINE ANIMATIONS ─────────── */
@keyframes nodePop {
    0%   { transform: scale(0.2); opacity: 0; }
    70%  { transform: scale(1.2); opacity: 0.9; }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes lineGrow {
    0%   { transform: scaleX(0); opacity: 0; }
    100% { transform: scaleX(1); opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 6px var(--green); }
    50%      { box-shadow: 0 0 16px var(--green), 0 0 24px rgba(34,197,94,0.7); }
}

/* ── WORKFLOW CARD (INTEGRATED PIPELINE DASHBOARD) ── */
.workflow-card {
    background: linear-gradient(180deg, #0d121c 0%, #070a10 100%);
    border: 1px solid rgba(37, 99, 235, 0.22);
    border-radius: var(--radius-md);
    padding: 14px 18px 12px;
    margin: 6px 0 12px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    animation: fadeInUp 0.5s ease;
}

.workflow-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 10px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.workflow-header-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.workflow-live-pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.7);
    animation: pulse-online 2s infinite;
}

.workflow-header-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue-glow);
}

.workflow-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 0.60rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.workflow-status-badge.success {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
}
.workflow-status-badge.warning {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

/* ── PIPELINE TRACK (6 COMPACT STAGES) ───── */
.pipeline-track {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    position: relative;
    padding: 4px 0 8px;
    gap: 0;
}

.pipeline-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    position: relative;
    z-index: 2;
    flex: 1;
    min-width: 0;
}

.stage-badge {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    margin-bottom: 6px;
    transition: var(--transition);
}

.stage-badge.completed {
    background: rgba(34, 197, 94, 0.16);
    border: 1.5px solid var(--green);
    color: var(--green);
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.4);
}

.stage-badge.running {
    background: rgba(37, 99, 235, 0.2);
    border: 1.5px solid var(--blue-bright);
    color: var(--blue-bright);
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.5);
    animation: pulseGlow 1.5s ease-in-out infinite;
}

.stage-badge.skipped {
    background: rgba(100, 116, 139, 0.15);
    border: 1.5px solid rgba(100, 116, 139, 0.4);
    color: var(--text-muted);
}

.stage-badge.failed {
    background: rgba(239, 68, 68, 0.15);
    border: 1.5px solid var(--red);
    color: var(--red);
    box-shadow: 0 0 10px rgba(239, 68, 68, 0.35);
}

.stage-connector {
    flex: 1;
    height: 2px;
    margin-top: 12px;
    min-width: 14px;
    background: rgba(255, 255, 255, 0.1);
    z-index: 1;
    border-radius: 1px;
    transition: var(--transition);
}

.stage-connector.completed,
.stage-connector.active {
    background: linear-gradient(90deg, var(--green), var(--blue-bright));
    box-shadow: 0 0 6px rgba(34, 197, 94, 0.4);
}

.stage-connector.failed {
    background: linear-gradient(90deg, var(--green), var(--red));
}

.stage-name {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin-bottom: 3px;
    white-space: nowrap;
}

.stage-value {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    padding: 2px 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 95%;
}

/* ── WORKFLOW FOOTER METRICS & PILLS ──────── */
.workflow-footer-strip {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.wf-stat-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    padding: 3px 8px;
}

.wf-stat-label {
    font-family: var(--font-mono);
    font-size: 0.52rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.wf-stat-value {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 600;
    color: var(--blue-glow);
}

.wf-stat-spacer {
    flex: 1;
}

.wf-pill-group {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Legacy pipeline compatibility styles */
.pipeline-container {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 10px 0;
    margin: 8px 0;
    overflow-x: auto;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 80px;
    flex-shrink: 0;
}
.pipeline-connector {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, var(--green), var(--blue-primary));
    min-width: 16px;
    margin-bottom: 24px;
}
.pipeline-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--blue-primary);
    margin-bottom: 6px;
}
.pipeline-dot.completed { background: var(--green); box-shadow: 0 0 8px rgba(34,197,94,0.4); }
.pipeline-dot.failed { background: var(--red); }
.pipeline-dot.skipped { background: var(--text-muted); }
.pipeline-dot.running { background: var(--blue-primary); animation: pulseGlow 1.5s infinite; }
.pipeline-label { font-family: var(--font-mono); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); }
.pipeline-value { font-family: var(--font-mono); font-size: 0.60rem; color: var(--text-secondary); margin-top: 2px; }

/* ── RESULT CARD ──────────────────────────── */
.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}
.result-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
}

.answer-surface {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px 20px 12px;
    margin: 6px 0 8px;
    animation: fadeInUp 0.5s ease;
}
.answer-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue-primary);
    margin-bottom: 4px;
}
.answer-grounded {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--green);
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 3px;
    padding: 3px 8px;
    margin-bottom: 0;
}
.answer-text {
    font-family: var(--font-sans);
    font-size: 0.95rem;
    color: var(--text-primary);
    line-height: 1.7;
    margin-top: 14px;
}

/* ── META PILLS ───────────────────────────── */
.meta-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.pill {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 3px;
    text-transform: uppercase;
}
.pill-intent   { background: var(--blue-dim-2); border: 1px solid var(--border-blue); color: var(--blue-glow); }
.pill-high     { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25); color: #4ade80; }
.pill-medium   { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); color: #fbbf24; }
.pill-low      { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); color: #f87171; }
.pill-sources  { background: var(--surface-3); border: 1px solid var(--border-mid); color: var(--text-secondary); }

/* ── NUMBERED FINDINGS ────────────────────── */
.findings-container {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 8px 16px;
    margin-bottom: 10px;
}
.finding-row {
    display: flex;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
    animation: fadeInUp 0.4s ease;
}
.finding-row:last-child { border-bottom: none; }
.finding-num {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--blue-primary);
    min-width: 24px;
    flex-shrink: 0;
    padding-top: 1px;
}
.finding-text {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--text-primary);
    line-height: 1.55;
}

/* ── READABLE LEGAL ANSWER STYLING ────────── */
.readable-answer-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 20px 22px;
    margin: 6px 0 14px;
    font-family: var(--font-sans);
    font-size: 0.92rem;
    line-height: 1.75;
    color: var(--text-primary);
}

.evidence-source-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.14) 0%, rgba(37, 99, 235, 0.04) 100%);
    border: 1px solid rgba(59, 130, 246, 0.28);
    border-left: 3px solid var(--blue-bright);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 16px 0 10px;
}
.evidence-source-card-header:first-child {
    margin-top: 2px;
}

.evidence-source-icon {
    font-size: 0.95rem;
}

.evidence-source-doc {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--blue-glow);
    letter-spacing: 0.04em;
}

.evidence-source-page {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 700;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #4ade80;
    border-radius: 4px;
    padding: 2px 7px;
    letter-spacing: 0.06em;
    margin-left: auto;
}

.statute-pill {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 700;
    color: #fbbf24;
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.32);
    border-radius: 4px;
    padding: 1px 6px;
    margin: 0 2px;
}

.timestamp-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-mono);
    font-size: 0.70rem;
    font-weight: 700;
    color: #67e8f9;
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 4px 4px 4px 0;
}

.timestamp-date {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    font-style: italic;
}

.timeline-ts-row {
    margin: 4px 0 2px;
}

.evidence-quote {
    background: rgba(13, 17, 24, 0.75);
    border-left: 3px solid var(--blue-primary);
    border-radius: 0 6px 6px 0;
    padding: 10px 16px;
    margin: 10px 0 12px;
    font-style: italic;
    color: #f1f5f9;
    line-height: 1.65;
}

.qa-badge-q {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--blue-glow);
    background: rgba(37, 99, 235, 0.14);
    border: 1px solid rgba(37, 99, 235, 0.28);
    border-radius: 3px;
    padding: 1px 5px;
    margin-right: 4px;
}

.qa-badge-a {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    font-weight: 700;
    color: #4ade80;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.28);
    border-radius: 3px;
    padding: 1px 5px;
    margin-right: 4px;
}

.finding-source-tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--blue-glow);
    background: rgba(37, 99, 235, 0.14);
    border: 1px solid rgba(59, 130, 246, 0.28);
    border-radius: 4px;
    padding: 1px 6px;
    margin-right: 4px;
}

.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(59, 130, 246, 0.38);
    border-radius: 4px;
    padding: 2px 7px;
    margin: 1px 3px;
    vertical-align: middle;
    font-size: 0.80rem;
    line-height: 1.35;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    white-space: nowrap;
}

.citation-pill:hover {
    border-color: rgba(96, 165, 250, 0.65);
    background: rgba(30, 41, 59, 0.98);
}

.citation-icon {
    font-size: 0.85rem;
    opacity: 0.9;
}

.citation-doc {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--blue-glow);
    letter-spacing: 0.02em;
}

.citation-divider {
    color: var(--text-muted);
    opacity: 0.45;
    font-size: 0.70rem;
    margin: 0 1px;
}

.citation-page {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 700;
    color: #4ade80;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 3px;
    padding: 1px 5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.passage-badge {
    display: inline-flex;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    color: #c084fc;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 3px;
    padding: 2px 6px;
    margin-right: 5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── SOURCE CARD ──────────────────────────── */
.source-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: var(--transition);
}
.source-card:hover {
    border-color: var(--border-blue);
    box-shadow: 0 2px 12px rgba(37,99,235,0.08);
}
.source-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.source-id {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue-primary);
}
.source-page {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.source-name {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
}
.source-excerpt {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.6;
    border-left: 2px solid var(--border-blue);
    padding-left: 12px;
    font-style: italic;
}
.source-meta {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
}
.source-meta-item {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── HISTORY ITEM ─────────────────────────── */
.history-item {
    display: flex;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid var(--border-subtle);
    transition: var(--transition);
    cursor: pointer;
}
.history-item:hover {
    background: var(--surface-1);
    margin: 0 -16px;
    padding: 14px 16px;
    border-radius: var(--radius-sm);
}
.history-time {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    min-width: 48px;
    flex-shrink: 0;
    padding-top: 2px;
}
.history-content {
    flex: 1;
    min-width: 0;
}
.history-intent {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue-primary);
    margin-bottom: 3px;
}
.history-query {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── DOCUMENT ROW ─────────────────────────── */
.doc-card {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: var(--transition);
}
.doc-card:hover {
    border-color: var(--border-blue);
}
.doc-type {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--blue-primary);
    background: var(--blue-dim);
    border: 1px solid var(--border-blue);
    border-radius: 3px;
    padding: 4px 8px;
    min-width: 36px;
    text-align: center;
}
.doc-name {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-primary);
    flex: 1;
}
.doc-status {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--green);
}

/* ── SIDEBAR CUSTOM ───────────────────────── */
.sb-brand {
    padding: 8px 0 4px;
    margin-bottom: 2px;
}
.sb-brand-name {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--blue-primary);
}
.sb-brand-title {
    font-family: var(--font-sans);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 2px;
}
.sb-section {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.sb-stat-label {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.sb-stat-value {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--blue-bright);
}
.sb-status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
}
.sb-status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-online 2.5s ease-in-out infinite;
}
.sb-status-text {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--green);
    font-weight: 600;
}

/* ── WORKFLOW LOG STEPS ───────────────────── */
.wf-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
    font-family: var(--font-mono);
    font-size: 0.75rem;
}
.wf-step:last-child { border-bottom: none; }
.wf-num {
    color: var(--blue-primary);
    font-weight: 600;
    min-width: 22px;
    flex-shrink: 0;
}
.wf-name {
    font-weight: 600;
    color: var(--text-primary);
}
.wf-detail {
    color: var(--text-secondary);
    font-weight: 400;
}

/* ── EMPTY STATE ──────────────────────────── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    animation: fadeInUp 0.5s ease;
}
.empty-icon {
    font-size: 2rem;
    color: var(--text-muted);
    margin-bottom: 12px;
    opacity: 0.5;
}
.empty-title {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.empty-desc {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    color: var(--text-muted);
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── GRID BACKGROUND ──────────────────────── */
.grid-bg {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(37,99,235,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(37,99,235,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: grid-drift 20s linear infinite;
    opacity: 0.5;
}

/* ── BALANCED SPACING & LAYOUT (NO OVERLAPS) ── */
[data-testid="stVerticalBlock"] {
    gap: 0.8rem !important;
}

[data-testid="stHorizontalBlock"] {
    gap: 0.65rem !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.65rem !important;
}

[data-testid="stFileUploader"] {
    margin-bottom: 12px !important;
}

[data-testid="stSidebar"] .stButton {
    margin-bottom: 4px !important;
}

[data-testid="stMetric"] {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 12px !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.5rem !important;
}

</style>
""", unsafe_allow_html=True)

# Subtle grid background
st.markdown('<div class="grid-bg"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE (EXPERIMENT 4)
# ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager(VECTORSTORE_DIR)
if "embedding_provider" not in st.session_state:
    st.session_state.embedding_provider = "local"
if "api_key" not in st.session_state:
    st.session_state.api_key = GEMINI_API_KEY or OPENAI_API_KEY
if "llm_provider" not in st.session_state:
    if DEFAULT_LLM_PROVIDER == "ollama":
        is_ollama_alive, _ = LLMService.check_ollama_status(OLLAMA_MODEL)
        st.session_state.llm_provider = "ollama" if is_ollama_alive else ("gemini" if GEMINI_API_KEY else "local")
    else:
        st.session_state.llm_provider = DEFAULT_LLM_PROVIDER

if "llm_model" not in st.session_state:
    if st.session_state.llm_provider == "gemini":
        st.session_state.llm_model = DEFAULT_GEMINI_MODEL
    elif st.session_state.llm_provider == "openai":
        st.session_state.llm_model = "gpt-4o-mini"
    elif st.session_state.llm_provider == "ollama":
        st.session_state.llm_model = OLLAMA_MODEL
    else:
        st.session_state.llm_model = "local-rag-fallback"

# Automatically migrate deprecated or non-existent models from existing browser session
if st.session_state.get("llm_model") in ["gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-3.8-flash", "models/gemini-3.8-flash"]:
    st.session_state.llm_model = "gemini-2.0-flash"
if "history" not in st.session_state:
    st.session_state.history = []
if "preset_query" not in st.session_state:
    st.session_state.preset_query = ""

# ─────────────────────────────────────────────
# INSTANTIATE SERVICES
# ─────────────────────────────────────────────
embedding_service = EmbeddingService(
    provider=st.session_state.embedding_provider,
    api_key=st.session_state.api_key
)
retriever = RAGRetriever(st.session_state.vector_store, embedding_service)
llm_service = LLMService(
    provider=st.session_state.llm_provider,
    model_name=st.session_state.llm_model,
    api_key=st.session_state.api_key,
    base_url=OLLAMA_BASE_URL
)
agent = LegalResearchAgent(retriever, llm_service)
stats = st.session_state.vector_store.get_stats()

# Auto-populate sample case documents if the knowledge base is empty on first launch
if stats.get("total_chunks", 0) == 0:
    try:
        sample_files = sorted(list(SAMPLE_DATA_DIR.glob("*.txt")))
        if sample_files:
            splitter = LegalTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
            for sf in sample_files:
                pages_data = DocumentLoader.load_file(sf, sf.name)
                chunks = splitter.split_pages(pages_data)
                st.session_state.vector_store.add_chunks(chunks, embedding_service)
            stats = st.session_state.vector_store.get_stats()
    except Exception:
        pass

# ─────────────────────────────────────────────
# SIDEBAR — NAVIGATION & CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-name">CSI RAG</div>
        <div class="sb-brand-title">Legal Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Knowledge Base section
    st.markdown('<div class="sb-section">Knowledge Base</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Import Case Evidence",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col_proc, col_sample = st.columns(2)
    with col_proc:
        if st.button("Process", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Indexing..."):
                    total_chunks_added = 0
                    splitter = LegalTextSplitter(
                        chunk_size=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                        chunk_overlap=st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                    )
                    for file in uploaded_files:
                        try:
                            file_path = DOCUMENTS_DIR / file.name
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            pages_data = DocumentLoader.load_file(file_path, file.name)
                            chunks = splitter.split_pages(pages_data)
                            st.session_state.vector_store.add_chunks(chunks, embedding_service)
                            total_chunks_added += len(chunks)
                        except Exception as e:
                            st.error(f"Error: {file.name}: {str(e)}")
                    st.success(f"Indexed {len(uploaded_files)} file(s), {total_chunks_added} chunks")
                    st.rerun()
            else:
                st.warning("Select files first.")

    with col_sample:
        if st.button("Samples", use_container_width=True):
            with st.spinner("Loading sample case files..."):
                try:
                    sample_files = sorted(list(SAMPLE_DATA_DIR.glob("*.txt")))
                    splitter = LegalTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
                    total_chunks = 0
                    for sf in sample_files:
                        pages_data = DocumentLoader.load_file(sf, sf.name)
                        chunks = splitter.split_pages(pages_data)
                        st.session_state.vector_store.add_chunks(chunks, embedding_service)
                        total_chunks += len(chunks)
                    st.success(f"Loaded {len(sample_files)} samples, {total_chunks} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load sample files: {str(e)}")

    if st.button("Clear Knowledge Base", use_container_width=True):
        st.session_state.vector_store.clear_store()
        st.success("Knowledge base cleared.")
        st.rerun()

    # Model Configuration
    st.markdown('<div class="sb-section">LLM Engine</div>', unsafe_allow_html=True)
    provider_choice = st.selectbox(
        "Provider",
        options=["ollama", "gemini", "openai", "local"],
        index=["ollama", "gemini", "openai", "local"].index(st.session_state.llm_provider),
        label_visibility="collapsed"
    )
    st.session_state.llm_provider = provider_choice

    if provider_choice == "ollama":
        model_choice = st.text_input("Model", value=OLLAMA_MODEL, label_visibility="collapsed", placeholder="Model name...")
        api_key_input = ""
        is_alive, ollama_msg = LLMService.check_ollama_status(model_choice)
        if ollama_msg:
            if is_alive:
                st.caption(f"✓ {ollama_msg} (Context capped at 2k to prevent heating)")
            else:
                st.caption(f"⚠️ {ollama_msg} — Switch to 'gemini' or 'local' on Streamlit Cloud")
    elif provider_choice == "gemini":
        gemini_model_options = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "Custom..."
        ]
        curr = st.session_state.get("llm_model", "gemini-2.0-flash")
        if not curr or curr in ["gemini-2.5-flash", "models/gemini-2.5-flash", "gemini-3.8-flash", OLLAMA_MODEL]:
            curr = "gemini-2.0-flash"
        idx = gemini_model_options.index(curr) if curr in gemini_model_options else 0
        sel_model = st.selectbox("Model", options=gemini_model_options, index=idx, label_visibility="collapsed")
        if sel_model == "Custom...":
            model_choice = st.text_input("Custom Model", value=(curr if curr not in gemini_model_options else "gemini-2.0-flash"), label_visibility="collapsed")
        else:
            model_choice = sel_model

        api_key_input = st.text_input("API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed", placeholder="Enter Gemini API Key...")
        is_alive = True
        ollama_msg = ""
        if api_key_input:
            st.caption("✓ Google Cloud active (1-2s cloud inference, 0% local heat)")
        else:
            st.caption("⚠️ Gemini requires an API key — or switch to 'local' mode")
    elif provider_choice == "openai":
        model_choice = st.text_input("Model", value="gpt-4o-mini", label_visibility="collapsed")
        api_key_input = st.text_input("API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed", placeholder="API Key...")
        is_alive = True
        ollama_msg = ""
    else:
        model_choice = "local-rag-fallback"
        api_key_input = ""
        is_alive = True
        ollama_msg = ""
        st.caption("⚡ Fast deterministic local RAG (Instant <0.1s, 0% CPU heat)")

    st.session_state.llm_model = model_choice
    st.session_state.api_key = api_key_input

    # RAG Settings
    st.markdown('<div class="sb-section">Retrieval Config</div>', unsafe_allow_html=True)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=st.session_state.get("top_k", DEFAULT_TOP_K), label_visibility="collapsed")
    st.session_state.top_k = top_k

    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE), step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=400, value=st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP), step=25)
        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

    # Session History controls
    st.markdown('<div class="sb-section">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-stat"><span class="sb-stat-label">Queries</span><span class="sb-stat-value">{len(st.session_state.history)}</span></div>
    """, unsafe_allow_html=True)
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.success("History cleared.")
        st.rerun()

    # System Status
    st.markdown('<div class="sb-section">System Status</div>', unsafe_allow_html=True)

    if provider_choice == "ollama" and is_alive:
        st.markdown("""
        <div class="sb-status-row">
            <div class="sb-status-dot"></div>
            <span class="sb-status-text">Online</span>
        </div>
        """, unsafe_allow_html=True)
    elif provider_choice == "ollama":
        st.markdown(f"""
        <div class="sb-status-row">
            <div class="sb-status-dot" style="background:var(--amber);animation:none;"></div>
            <span class="sb-status-text" style="color:var(--amber);">Offline</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-status-row">
            <div class="sb-status-dot"></div>
            <span class="sb-status-text">Ready</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-stat"><span class="sb-stat-label">LLM</span><span class="sb-stat-value">{provider_choice.upper()}</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Model</span><span class="sb-stat-value" style="font-size:0.68rem;">{model_choice}</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Documents</span><span class="sb-stat-value">{stats['total_documents']:02d}</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Chunks</span><span class="sb-stat-value">{stats['total_chunks']}</span></div>
    <div class="sb-stat"><span class="sb-stat-label">Top-K</span><span class="sb-stat-value">{top_k}</span></div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SYNCHRONIZE ACTIVE SERVICES WITH SIDEBAR SELECTIONS
# ─────────────────────────────────────────────
embedding_service = EmbeddingService(
    provider=st.session_state.embedding_provider,
    api_key=st.session_state.api_key
)
retriever = RAGRetriever(st.session_state.vector_store, embedding_service)
llm_service = LLMService(
    provider=st.session_state.llm_provider,
    model_name=st.session_state.llm_model,
    api_key=st.session_state.api_key,
    base_url=OLLAMA_BASE_URL
)
agent = LegalResearchAgent(retriever, llm_service)

# ─────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────
now_time = datetime.now().strftime("%H:%M")
ollama_status_label = "ONLINE" if (provider_choice == "ollama" and is_alive) else ("OFFLINE" if provider_choice == "ollama" else "READY")
ollama_dot_class = "online" if ollama_status_label in ("ONLINE", "READY") else ""

st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left"><strong>Case Intelligence</strong> &mdash; Legal Research Terminal</div>
    <div class="top-bar-right">
        <div class="top-bar-status">
            {provider_choice.upper()}
            <span class="status-dot {"online" if ollama_status_label != "OFFLINE" else ""}" style="{"background:var(--amber);animation:none;" if ollama_status_label == "OFFLINE" else ""}"></span>
            {ollama_status_label}
        </div>
        <div class="top-bar-status">
            Vector DB
            <span class="status-dot ready"></span>
            READY
        </div>
        <div class="top-bar-status" style="color:var(--text-secondary);">{now_time}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Legal Intelligence Platform</div>
    <div class="hero-title">Crime Scene <span class="highlight">Investigation</span> RAG</div>
</div>
""", unsafe_allow_html=True)

# System notice
st.markdown("""
<div class="system-notice">
    <div class="notice-indicator"></div>
    <div class="notice-content">
        <div class="notice-label">Research Mode Active</div>
        <div class="notice-text">This system provides AI-assisted legal research based solely on documents in the knowledge base. It does not constitute legal advice and must not substitute professional legal counsel.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_research, tab_history, tab_documents, tab_evaluation = st.tabs(["RESEARCH", "HISTORY", "CASE FILES", "RAG EVALUATION"])

# ═══════════════════════════════════════════════
# TAB 1 — RESEARCH
# ═══════════════════════════════════════════════
with tab_research:

    # Metrics bar
    vec_status = "READY" if stats['total_chunks'] > 0 else "EMPTY"
    st.markdown(f"""
    <div class="metrics-bar">
        <div class="metric-card">
            <div class="metric-value">{stats['total_documents']:02d}</div>
            <div class="metric-label">Documents</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['total_chunks']}</div>
            <div class="metric-label">Indexed Chunks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:{"var(--green)" if vec_status == "READY" else "var(--amber)"};font-size:0.9rem;">{vec_status}</div>
            <div class="metric-label">Vector Index</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:0.9rem;">{provider_choice.upper()}</div>
            <div class="metric-label">LLM Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick prompts
    st.markdown('<div class="section-label">Quick Research Prompts</div>', unsafe_allow_html=True)

    preset_map = {
        "Case Summary":       ("Reconstruct case facts", "What happened in this case?"),
        "Evidence Analysis":  ("Examine evidence & sources", "What evidence connects the accused to the crime?"),
        "Case Timeline":      ("Sequence of events", "Give me the timeline of events."),
        "Contradictions":     ("Find inconsistencies", "Are there contradictions between the witness statements?"),
        "Hallucination Test": ("Out-of-scope query", "Who won the 2024 FIFA World Cup?"),
    }
    preset_keys = list(preset_map.keys())
    cols = st.columns(5)
    for i, (col, key) in enumerate(zip(cols, preset_keys)):
        with col:
            desc, query = preset_map[key]
            st.markdown(f"""
            <div class="prompt-card" id="prompt-{i}">
                <div class="prompt-num">{i+1:02d}</div>
                <div class="prompt-title">{key}</div>
                <div class="prompt-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(key, use_container_width=True, key=f"preset_{i}"):
                st.session_state.preset_query = query

    # Command input
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label-blue">Research Query</div>', unsafe_allow_html=True)

    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Query",
            value=st.session_state.preset_query,
            placeholder="Ask the case intelligence system...",
            label_visibility="collapsed"
        )
        submit_btn = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    # Meta tags below input
    top_k_val = st.session_state.get("top_k", DEFAULT_TOP_K)
    st.markdown(f"""
    <div class="command-meta">
        <span class="command-tag active">RAG Active</span>
        <span class="command-tag">Top-K {top_k_val}</span>
        <span class="command-tag">{provider_choice.upper()} / {model_choice}</span>
    </div>
    """, unsafe_allow_html=True)

    # ─── EXECUTE RESEARCH ─────────────────────
    if submit_btn:
        if not user_query or not user_query.strip():
            st.warning("Please enter a valid research question.")
        elif stats['total_chunks'] == 0 and not is_standalone_hypothetical_query(user_query):
            st.warning("Knowledge base is empty. Upload documents or load samples from the sidebar, or ask a clearly self-contained hypothetical question.")
        else:
            st.session_state.preset_query = user_query
            # Live workflow progress: update the status as real pipeline stages finish.
            with st.status("AGENT WORKFLOW IN PROGRESS...", expanded=True) as status_container:
                progress_line = st.empty()
                progress_line.write("Starting research workflow...")

                def show_workflow_progress(message: str) -> None:
                    progress_line.write(f"● {message}")

                state = None
                try:
                    try:
                        state = agent.run_workflow(
                            query=user_query,
                            top_k=top_k_val,
                            progress_callback=show_workflow_progress,
                        )
                    except TypeError as type_err:
                        if "progress_callback" in str(type_err):
                            state = agent.run_workflow(
                                query=user_query,
                                top_k=top_k_val,
                            )
                        else:
                            raise
                except Exception as exc:
                    status_container.update(
                        label="Research workflow failed",
                        state="error",
                        expanded=True,
                    )
                    st.error(f"Research failed safely: {type(exc).__name__}: {exc}")

                if state is not None:
                    if state.status == "Completed":
                        status_container.update(
                            label="✓ Analysis Complete",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status_container.update(
                            label="Research stopped safely — see the error details below",
                            state="error",
                            expanded=True,
                        )
                        if state.response:
                            st.error(state.response)

                    # Store in history
                    history_item = {
                        "query": user_query,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "state": state,
                        "answer_data": state.structured_answer.model_dump() if state.structured_answer else {}
                    }
                    st.session_state.history.insert(0, history_item)

    # ─── RENDER RESULTS ──────────────────────
    if st.session_state.history:
        latest = st.session_state.history[0]
        state: WorkflowState = latest["state"]
        ans: LegalResearchAnswer = state.structured_answer

        strength = state.evidence_strength
        strength_class = "pill-high" if strength == "High" else ("pill-medium" if strength == "Medium" else "pill-low")
        intent_val = state.intent.value if hasattr(state.intent, 'value') else state.intent

        # Validation notice
        if state.validation_error:
            if any(term in state.validation_error.lower() for term in ["api notice", "api invocation", "ollama error", "service temporarily"]):
                st.warning(f"Provider Notice: {state.validation_error}")
            else:
                st.warning(f"Structured output validation: {state.validation_error}")

        # Runtime metrics are measured by the workflow, not estimated.
        runtime_metrics = getattr(state, "metrics", {}) or {}

        # Workflow Pipeline Visualization (Sequential Load Animation with real states)
        retrieval_status = "completed" if len(state.retrieved_chunks) > 0 else ("skipped" if is_standalone_hypothetical_query(state.query) else "failed")
        evidence_status = "completed" if strength in {"High", "Medium"} else ("skipped" if strength == "User-provided hypothetical" else "failed" if strength == "Insufficient" else "completed")
        specialist_count = int(runtime_metrics.get("specialist_agents_completed", 0))
        specialist_status = "completed" if specialist_count > 0 else ("skipped" if (strength == "Insufficient" or is_standalone_hypothetical_query(state.query)) else "failed")
        answer_status = "completed" if (ans and state.status == "Completed") else ("failed" if state.status == "Failed" else "skipped")

        unique_cited_docs = len({src['document_name'] for src in state.sources if src.get('document_name')})
        cited_chunks_cnt = len(state.sources)
        retrieved_chunk_cnt = int(runtime_metrics.get("retrieved_chunk_count", len(state.retrieved_chunks)))
        retrieved_doc_cnt = int(runtime_metrics.get("retrieved_distinct_document_count", len({c.document_name for c in state.retrieved_chunks})))

        specialist_selected = int(runtime_metrics.get("specialist_agents_selected", specialist_count))
        status_text = "Analysis Complete" if state.status == "Completed" else state.status
        status_badge_class = "success" if state.status == "Completed" else "warning"
        status_icon = "&#10003;" if state.status == "Completed" else "&#9679;"

        st.html(f"""
        <div class="workflow-card">
            <div class="workflow-header-row">
                <div class="workflow-header-left">
                    <span class="workflow-live-pulse"></span>
                    <span class="workflow-header-title">Agent Workflow Pipeline</span>
                </div>
                <div class="workflow-header-right">
                    <span class="workflow-status-badge {status_badge_class}">
                        {status_icon} {status_text} &bull; {runtime_metrics.get('total_duration_seconds', 0):.2f}s
                    </span>
                </div>
            </div>

            <div class="pipeline-track">
                <!-- Step 1: Query -->
                <div class="pipeline-stage">
                    <div class="stage-badge completed">&#10003;</div>
                    <div class="stage-name">Query</div>
                    <div class="stage-value">Received</div>
                </div>

                <div class="stage-connector active"></div>

                <!-- Step 2: Intent -->
                <div class="pipeline-stage">
                    <div class="stage-badge completed">&#10003;</div>
                    <div class="stage-name">Intent</div>
                    <div class="stage-value">{html.escape(intent_val.replace('_', ' ').title())}</div>
                </div>

                <div class="stage-connector {retrieval_status}"></div>

                <!-- Step 3: Retrieval -->
                <div class="pipeline-stage">
                    <div class="stage-badge {retrieval_status}">{"&#10003;" if retrieval_status == "completed" else "!"}</div>
                    <div class="stage-name">Retrieval</div>
                    <div class="stage-value">{unique_cited_docs} doc(s) ({cited_chunks_cnt} cited)</div>
                </div>

                <div class="stage-connector {evidence_status}"></div>

                <!-- Step 4: Evidence -->
                <div class="pipeline-stage">
                    <div class="stage-badge {evidence_status}">{"&#10003;" if evidence_status == "completed" else "!"}</div>
                    <div class="stage-name">Evidence</div>
                    <div class="stage-value">{html.escape(strength)}</div>
                </div>

                <div class="stage-connector {specialist_status}"></div>

                <!-- Step 5: Specialists -->
                <div class="pipeline-stage">
                    <div class="stage-badge {specialist_status}">{"&#10003;" if specialist_status == "completed" else "!"}</div>
                    <div class="stage-name">Specialists</div>
                    <div class="stage-value">{specialist_count} active</div>
                </div>

                <div class="stage-connector {answer_status}"></div>

                <!-- Step 6: Answer -->
                <div class="pipeline-stage">
                    <div class="stage-badge {answer_status}">{"&#10003;" if answer_status == "completed" else "!"}</div>
                    <div class="stage-name">Answer</div>
                    <div class="stage-value">{"Validated" if ans else "Raw"}</div>
                </div>
            </div>

            <div class="workflow-footer-strip">
                <div class="wf-stat-chip">
                    <span class="wf-stat-label">RUNTIME</span>
                    <span class="wf-stat-value">{runtime_metrics.get('total_duration_seconds', 0):.2f}s</span>
                </div>
                <div class="wf-stat-chip">
                    <span class="wf-stat-label">RETRIEVED</span>
                    <span class="wf-stat-value">{retrieved_chunk_cnt} chunks ({retrieved_doc_cnt} docs)</span>
                </div>
                <div class="wf-stat-chip">
                    <span class="wf-stat-label">AGENTS</span>
                    <span class="wf-stat-value">{specialist_count}/{specialist_selected} completed</span>
                </div>
                <div class="wf-stat-chip">
                    <span class="wf-stat-label">CITATIONS</span>
                    <span class="wf-stat-value">{cited_chunks_cnt} cited ({unique_cited_docs} docs)</span>
                </div>
                <div class="wf-stat-spacer"></div>
                <div class="wf-pill-group">
                    <span class="pill pill-intent">Intent: {html.escape(intent_val.replace('_', ' '))}</span>
                    <span class="pill {strength_class}">Evidence: {html.escape(strength)}</span>
                </div>
            </div>
        </div>
        """)

        # ─── ANSWER CARD ─────────────────────
        if ans:
            norm_disp_html = ""
            if getattr(state, 'normalized_query', None) and state.normalized_query.strip().lower() != state.query.strip().lower():
                norm_disp_html = f'<div style="font-size:0.82rem;color:var(--blue-glow);margin-top:4px;font-family:var(--font-mono);"><span style="color:var(--text-muted);">Normalized:</span> {html.escape(state.normalized_query)}</div>'

            clean_query = html.escape(state.query)
            formatted_answer = format_readable_answer(ans.answer)
            st.html(f"""
            <div class="answer-surface" style="padding-bottom: 8px;">
                <div class="result-header">
                    <div>
                        <div class="answer-label">AI Research Response</div>
                        <div style="font-size:1.05rem;font-weight:600;color:var(--text-primary);margin-top:4px;">{clean_query}</div>
                        {norm_disp_html}
                    </div>
                    <div class="answer-grounded">
                        <span style="width:5px;height:5px;border-radius:50%;background:var(--green);display:inline-block;"></span>
                        Source Grounded
                    </div>
                </div>
            </div>
            """)
            st.markdown(
                f'<div class="readable-answer-card">{formatted_answer}</div>',
                unsafe_allow_html=True
            )

            # Key Findings + Evidence in columns
            col_findings, col_evidence = st.columns(2)

            with col_findings:
                st.html('<div class="section-label-blue">Key Legal Findings</div>')
                findings_html = '<div class="findings-container">'
                for idx, f in enumerate(ans.key_findings, 1):
                    formatted_f = format_finding_item(f)
                    findings_html += f"""
                    <div class="finding-row">
                        <div class="finding-num">{idx:02d}</div>
                        <div class="finding-text">{formatted_f}</div>
                    </div>"""
                findings_html += '</div>'
                st.html(findings_html)

            with col_evidence:
                st.html('<div class="section-label-blue">Evidence & Witness Summary</div>')
                evidence_html = '<div class="findings-container">'
                for idx, e in enumerate(ans.evidence_summary, 1):
                    formatted_e = format_finding_item(e)
                    evidence_html += f"""
                    <div class="finding-row">
                        <div class="finding-num">{idx:02d}</div>
                        <div class="finding-text">{formatted_e}</div>
                    </div>"""
                evidence_html += '</div>'
                st.html(evidence_html)

            # Limitations
            if ans.limitations:
                st.html('<div class="section-label">Limitations & Unknowns</div>')
                lim_html = '<div class="findings-container">'
                for idx, lim in enumerate(ans.limitations, 1):
                    formatted_l = format_finding_item(lim)
                    lim_html += f"""
                    <div class="finding-row">
                        <div class="finding-num" style="color:var(--text-muted);">{idx:02d}</div>
                        <div class="finding-text" style="color:var(--text-secondary);">{formatted_l}</div>
                    </div>"""
                lim_html += '</div>'
                st.html(lim_html)

        else:
            # Fallback for non-structured answer
            clean_query = html.escape(state.query)
            clean_fallback = format_readable_answer(state.final_answer or state.response or "")
            st.html(f"""
            <div class="answer-surface">
                <div class="answer-label">Research Response</div>
                <div style="margin-top:4px;font-size:1rem;font-weight:600;color:var(--text-primary);">{clean_query}</div>
            </div>
            """)
            st.markdown(
                f'<div class="readable-answer-card">{clean_fallback}</div>',
                unsafe_allow_html=True
            )

        # ─── SOURCES ─────────────────────────
        with st.expander(f"EVIDENCE SOURCES  [{unique_cited_docs} unique document(s), {cited_chunks_cnt} chunk(s) cited; {retrieved_chunk_cnt} retrieved]"):
            if not state.sources:
                st.html(f"""
                <div class="empty-state">
                    <div class="empty-title">No Citable Sources Accepted</div>
                    <div class="empty-desc">{len(state.retrieved_chunks)} candidate passage(s) were retrieved, but none passed the evidence sufficiency gate for citation.</div>
                </div>
                """)
            else:
                docs_to_sources = {}
                for src in state.sources:
                    doc = src.get("document_name", "Unknown Document")
                    docs_to_sources.setdefault(doc, []).append(src)

                for doc_name, doc_sources in docs_to_sources.items():
                    st.html(f"""
                    <div style="margin-top:12px; margin-bottom:6px; font-weight:700; font-size:0.88rem; color:var(--blue-glow); display:flex; align-items:center; gap:8px;">
                        <span>📄 {html.escape(doc_name)}</span>
                        <span style="font-size:0.75rem; font-weight:400; color:var(--text-muted);">({len(doc_sources)} chunk(s) cited)</span>
                    </div>
                    """)
                    for src in doc_sources:
                        st.html(f"""
                        <div class="source-card">
                            <div class="source-top">
                                <span class="source-id">Citation #{src['citation_num']:02d}</span>
                                <span class="source-page">Page {src['page_number']}</span>
                            </div>
                            <div class="source-excerpt">"{html.escape(src['excerpt'])}"</div>
                            <div class="source-meta">
                                <span class="source-meta-item">Chunk #{src['chunk_id']}</span>
                            </div>
                        </div>
                        """)

        # ─── EXECUTED TOOLS LOG ───────────────
        if getattr(state, "executed_tools", None):
            with st.expander(f"EXECUTED AGENT TOOLS  [{len(state.executed_tools)} tools invoked]"):
                for tool_item in state.executed_tools:
                    st.html(f"""
                    <div class="source-card" style="border-left: 3px solid var(--blue-bright);">
                        <div class="source-top">
                            <span class="source-id" style="color:var(--blue-glow); font-weight:700;">Tool: {html.escape(tool_item.tool_name)}</span>
                            <span class="command-tag active">{html.escape(tool_item.status.upper())}</span>
                        </div>
                        <div class="source-excerpt" style="margin-top:6px; font-family:var(--font-mono); font-size:0.75rem; white-space:pre-wrap;">{html.escape(tool_item.result_preview)}</div>
                    </div>
                    """)

        # ─── QUERY & RETRIEVAL DIAGNOSTICS ────
        with st.expander("QUERY UNDERSTANDING & RETRIEVAL DIAGNOSTICS"):
            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                st.markdown('<div class="section-label-blue">Query Preprocessing & Intent</div>', unsafe_allow_html=True)
                st.markdown(f"**Original Query:** `{state.query}`")
                st.markdown(f"**Normalized Query:** `{state.normalized_query or state.query}`")
                conf_val = getattr(state, 'intent_confidence', 1.0)
                st.markdown(f"**Classified Intent:** `{intent_val}` ({conf_val:.0%} confidence)")
                
                if getattr(state, 'normalization_notes', None):
                    st.markdown("**Transformations Applied:**")
                    for note in state.normalization_notes:
                        st.markdown(f"• <span style='color:var(--text-secondary);font-size:0.85rem;'>{note}</span>", unsafe_allow_html=True)

                if getattr(state, 'is_ambiguous', False):
                    st.warning(f"Ambiguity Flagged: {state.clarification_prompt}")

            with col_diag2:
                st.markdown('<div class="section-label-blue">Pipeline Execution Status</div>', unsafe_allow_html=True)
                metrics = getattr(state, 'metrics', {}) or {}
                st.markdown(f"**Workflow Dispatch:** `{metrics.get('workflow_dispatch_status', 'standard_research')}`")
                req_op = metrics.get('required_operation', 'None')
                req_inv = "Yes (Completed)" if metrics.get('required_operation_invoked') else "No"
                st.markdown(f"**Required Operation:** `{req_op}` (Invoked: `{req_inv}`)")
                st.markdown(f"**Retrieval Match:** `{strength}`")
                distinct_docs = metrics.get('distinct_document_count', len(set(c.document_name for c in state.retrieved_chunks)))
                st.markdown(f"**Chunks Retrieved:** `{len(state.retrieved_chunks)} passages` ({distinct_docs} distinct documents)")
                executed_tools_str = ', '.join(t.tool_name for t in state.executed_tools) if state.executed_tools else 'None'
                st.markdown(f"**Tools Invoked:** `{len(state.executed_tools)} tools` ({executed_tools_str})")
                
                # Per-stage execution timings
                ret_t = metrics.get('retrieval_duration_seconds', 0)
                spec_t = metrics.get('specialist_duration_seconds', 0)
                tool_t = metrics.get('tool_duration_seconds', 0)
                syn_t = metrics.get('synthesis_duration_seconds', 0)
                tot_t = metrics.get('total_duration_seconds', 0)
                st.caption(f"Stage Timings — Retrieval: {ret_t:.2f}s | Specialist: {spec_t:.2f}s | Tools: {tool_t:.2f}s | Synthesis: {syn_t:.2f}s | Total: {tot_t:.2f}s")
                
                if getattr(state, 'validation_error', None):
                    st.error(f"Validation Failure: {state.validation_error}")
                else:
                    st.success(f"Pipeline Status: {state.status.upper()} (Pydantic Validation: PASSED)")

            if state.retrieved_chunks:
                st.html('<div class="section-label" style="margin-top:14px;">Retrieved Chunks & Relevance Scores</div>')
                for c_idx, c_item in enumerate(state.retrieved_chunks, 1):
                    st.html(f"""
                    <div class="source-card" style="margin-bottom:8px;">
                        <div class="source-top">
                            <span class="source-id">Chunk #{c_item.chunk_id}</span>
                            <span class="command-tag active">Hybrid Score: {c_item.score:.3f}</span>
                            <span class="source-page">{html.escape(c_item.document_name)} (Page {c_item.page_number})</span>
                        </div>
                        <div class="source-excerpt" style="margin-top:6px;">"{html.escape(c_item.text[:220].strip())}..."</div>
                    </div>
                    """)
                    with st.expander(f"Full retrieved passage — {c_item.document_name}, page {c_item.page_number}"):
                        st.text(c_item.text)

        # ─── WORKFLOW LOG ─────────────────────
        with st.expander("AGENT WORKFLOW LOG"):
            st.html('<div class="section-label">Research Plan</div>')
            for p_idx, plan_item in enumerate(state.plan, 1):
                st.html(f"""
                <div class="wf-step">
                    <span class="wf-num">{p_idx:02d}</span>
                    <span class="wf-detail">{html.escape(plan_item)}</span>
                </div>
                """)

            st.html('<div class="section-label" style="margin-top:16px;">Execution Log</div>')
            for step in state.steps:
                st.html(f"""
                <div class="wf-step">
                    <span class="wf-num">[+]</span>
                    <span class="wf-name">{html.escape(step.step_name)}</span>
                    <span class="wf-detail"> &mdash; {html.escape(step.detail or '')}</span>
                </div>
                """)

            st.markdown('<div class="section-label" style="margin-top:16px;">State JSON</div>', unsafe_allow_html=True)
            st.json(state.to_dict())

            if state.raw_model_response:
                st.markdown('<div class="section-label" style="margin-top:16px;">Raw Model Output</div>', unsafe_allow_html=True)
                st.code(state.raw_model_response, language="json")

    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">No Research Results</div>
            <div class="empty-desc">Use the quick prompts above or enter a custom query to begin investigating case documents.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — HISTORY
# ═══════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section-label-blue">Research History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">No History</div>
            <div class="empty-desc">Research queries and their results will appear here as you interact with the system.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metrics-bar">
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.history)}</div>
                <div class="metric-label">Total Queries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for idx, item in enumerate(st.session_state.history):
            h_state: WorkflowState = item["state"]
            h_ans = item["answer_data"]
            h_intent = h_state.intent.value if hasattr(h_state.intent, 'value') else h_state.intent

            interaction_num = len(st.session_state.history) - idx
            with st.expander(f"{item['timestamp']}  |  {h_intent.replace('_',' ').title()}  |  {str(item['query'])[:80]}"):
                # Meta
                st.markdown(f"""
                <div class="meta-pills">
                    <span class="pill pill-intent">Intent: {h_intent.replace('_', ' ')}</span>
                    <span class="pill pill-sources">Sources: {len(h_state.sources)}</span>
                </div>
                """, unsafe_allow_html=True)

                if h_ans:
                    # Structured answer
                    formatted_h_ans = format_readable_answer(str(h_ans.get('answer', '')))
                    st.markdown(f"""
                    <div class="answer-surface" style="margin-top:12px;">
                        <div class="answer-label">Response</div>
                    </div>
                    <div class="readable-answer-card" style="margin-top:8px;">{formatted_h_ans}</div>
                    """, unsafe_allow_html=True)

                    if h_ans.get("key_findings"):
                        st.markdown('<div class="section-label" style="margin-top:16px;">Key Findings</div>', unsafe_allow_html=True)
                        findings_html = ""
                        for f_idx, kf in enumerate(h_ans.get("key_findings"), 1):
                            formatted_kf = format_finding_item(kf)
                            findings_html += f"""
                            <div class="finding-row">
                                <div class="finding-num">{f_idx:02d}</div>
                                <div class="finding-text">{formatted_kf}</div>
                            </div>"""
                        st.markdown(findings_html, unsafe_allow_html=True)
                elif h_state.final_answer or h_state.response:
                    fallback_h_ans = format_readable_answer(h_state.final_answer or h_state.response or "")
                    st.markdown(f"""
                    <div class="answer-surface" style="margin-top:12px;">
                        <div class="answer-label">Response</div>
                    </div>
                    <div class="readable-answer-card" style="margin-top:8px;">{fallback_h_ans}</div>
                    """, unsafe_allow_html=True)

                # Sources
                if h_state.sources:
                    st.markdown('<div class="section-label" style="margin-top:16px;">Sources</div>', unsafe_allow_html=True)
                    st.json(h_state.sources)


# ═══════════════════════════════════════════════
# TAB 3 — CASE FILES
# ═══════════════════════════════════════════════
with tab_documents:
    stats = st.session_state.vector_store.get_stats()
    st.markdown('<div class="section-label-blue">Case Files</div>', unsafe_allow_html=True)

    if stats['total_documents'] == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-title">No Documents Indexed</div>
            <div class="empty-desc">Upload case documents or load sample data from the sidebar to begin building the knowledge base.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Stats bar
        st.markdown(f"""
        <div class="metrics-bar">
            <div class="metric-card">
                <div class="metric-value">{stats['total_documents']:02d}</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Total Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:var(--green);font-size:0.9rem;">INDEXED</div>
                <div class="metric-label">Status</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Document cards
        st.markdown('<div class="section-label">Document Registry</div>', unsafe_allow_html=True)
        for i, doc_name in enumerate(stats['documents_list'], 1):
            ext = Path(doc_name).suffix.replace(".", "").upper() if "." in doc_name else "DOC"
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-type">{ext}</div>
                <div class="doc-name">{html.escape(str(doc_name))}</div>
                <div class="doc-status">Ready</div>
            </div>
            """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════
# TAB 4 — RAG EVALUATION & BENCHMARKING
# ═══════════════════════════════════════════════
with tab_evaluation:
    st.markdown('<div class="section-label-blue">Retrieval Evaluation</div>', unsafe_allow_html=True)
    st.caption(
        "Enter a legal-case question and optional ground-truth document names. "
        "Relevance metrics are meaningful only when the labels reflect manually verified relevant records."
    )

    evaluation_query = st.text_input(
        "Evaluation query",
        placeholder="e.g. Compare the guard's statement with the CCTV exit log",
        key="rag_eval_query",
    )
    relevant_documents_input = st.text_input(
        "Relevant document filenames (comma-separated, optional)",
        placeholder="02_Witness_Statements.txt, 03_CCTV_Analysis_Report.txt",
        key="rag_eval_relevant_docs",
    )
    relevant_chunks_input = st.text_input(
        "Relevant chunk IDs (comma-separated, optional)",
        placeholder="chunk_id_1, chunk_id_2",
        key="rag_eval_relevant_chunks",
    )

    eval_col1, eval_col2, eval_col3 = st.columns(3)
    with eval_col1:
        run_retrieval_eval = st.button("Evaluate Retrieval", type="primary", use_container_width=True)
    with eval_col2:
        run_latency_benchmark = st.button("Compare LLM-only vs RAG", use_container_width=True)
    with eval_col3:
        run_sample_benchmark = st.button("Run Sample Benchmark", use_container_width=True)

    if run_sample_benchmark:
        try:
            with st.spinner("Running labelled synthetic-case benchmark..."):
                sample_report = benchmark_retrieval(
                    retriever, DEFAULT_LEGAL_EVALUATION_CASES, top_k=top_k
                )
            st.markdown("**Sample benchmark aggregate**")
            aggregate = sample_report["aggregate"]
            sample_cols = st.columns(4)
            sample_cols[0].metric("Mean Precision@K", f"{aggregate['precision_at_k']:.3f}")
            sample_cols[1].metric("Mean Recall@K", f"{aggregate['recall_at_k']:.3f}")
            sample_cols[2].metric("Mean F1@K", f"{aggregate['f1_at_k']:.3f}")
            sample_cols[3].metric("Mean MRR", f"{aggregate['reciprocal_rank']:.3f}")
            st.caption(
                f"{sample_report['query_count']} labelled queries · "
                f"mean retrieval time {aggregate['retrieval_time_seconds']:.4f} s"
            )
            st.dataframe(sample_report["results"], use_container_width=True)
            st.info(
                "The sample labels target the bundled synthetic case files. "
                "The out-of-scope hostel-fee question intentionally has no relevant source."
            )
        except Exception as exc:
            st.error(f"Sample benchmark failed: {type(exc).__name__}: {exc}")

    if run_retrieval_eval:
        if not evaluation_query.strip():
            st.warning("Enter a question before running retrieval evaluation.")
        elif not relevant_documents_input.strip() and not relevant_chunks_input.strip():
            st.warning("Add at least one verified relevant filename or chunk ID to calculate precision and recall.")
        else:
            try:
                with st.spinner("Retrieving and scoring evidence..."):
                    retrieval_started = time.perf_counter()
                    eval_chunks, eval_strength = retriever.retrieve_evidence(
                        evaluation_query.strip(), top_k=top_k
                    )
                    retrieval_elapsed = time.perf_counter() - retrieval_started
                    eval_metrics = evaluate_retrieval(
                        eval_chunks,
                        relevant_document_names=[
                            item.strip() for item in relevant_documents_input.split(",") if item.strip()
                        ],
                        relevant_chunk_ids=[
                            item.strip() for item in relevant_chunks_input.split(",") if item.strip()
                        ],
                        k=top_k,
                        retrieval_time_seconds=retrieval_elapsed,
                    )
                st.markdown("**Retrieval results**")
                st.caption(f"Evidence match rating: {eval_strength} · Retrieved {len(eval_chunks)} chunk(s)")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Precision@K", f"{eval_metrics['precision_at_k']:.3f}")
                metric_cols[1].metric("Recall@K", f"{eval_metrics['recall_at_k']:.3f}")
                metric_cols[2].metric("F1@K", f"{eval_metrics['f1_at_k']:.3f}")
                metric_cols[3].metric("MRR", f"{eval_metrics['reciprocal_rank']:.3f}")
                st.json(eval_metrics)
                with st.expander("Retrieved passages used in this evaluation", expanded=True):
                    if not eval_chunks:
                        st.info("No passages were retrieved.")
                    for rank, chunk in enumerate(eval_chunks, 1):
                        st.markdown(
                            f"**{rank}. {chunk.document_name} · page {chunk.page_number} · "
                            f"score {chunk.score:.3f}**"
                        )
                        st.code(chunk.text)
            except Exception as exc:
                st.error(f"Retrieval evaluation failed: {type(exc).__name__}: {exc}")

    if run_latency_benchmark:
        if not evaluation_query.strip():
            st.warning("Enter a question before running the latency comparison.")
        elif provider_choice == "local":
            st.warning("The deterministic local fallback cannot provide an LLM-only generation baseline. Select Ollama, Gemini, or OpenAI.")
        else:
            try:
                with st.spinner("Running direct LLM and RAG workflows..."):
                    comparison = compare_llm_only_vs_rag(
                        evaluation_query.strip(), retriever, llm_service, agent, top_k=top_k
                    )
                left, right = st.columns(2)
                with left:
                    st.markdown("**LLM-only baseline**")
                    st.metric("Total latency", f"{comparison['llm_only']['latency_seconds']:.2f} s")
                    st.write(comparison["llm_only"]["answer"])
                with right:
                    st.markdown("**RAG workflow**")
                    st.metric("Total latency", f"{comparison['rag']['latency_seconds']:.2f} s")
                    st.metric("Retrieval-only latency", f"{comparison['rag']['retrieval_latency_seconds']:.2f} s")
                    st.caption(
                        f"{comparison['rag']['retrieved_count']} passages retrieved · "
                        f"{comparison['rag']['source_count']} cited sources · "
                        f"evidence rating: {comparison['rag']['evidence_strength']}"
                    )
                    st.write(comparison["rag"]["answer"])
                st.info(comparison["measurement_note"])
                st.json({
                    "llm_only_latency_seconds": comparison["llm_only"]["latency_seconds"],
                    "rag_total_latency_seconds": comparison["rag"]["latency_seconds"],
                    "rag_retrieval_latency_seconds": comparison["rag"]["retrieval_latency_seconds"],
                    "rag_minus_llm_seconds": comparison["latency_delta_seconds"],
                    "rag_workflow_status": comparison["rag"]["workflow_status"],
                })
            except Exception as exc:
                st.error(f"Latency comparison failed: {type(exc).__name__}: {exc}")
