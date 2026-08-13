import os
from pathlib import Path
import streamlit as st
import json
import time
from datetime import datetime

from config import (
    DOCUMENTS_DIR, VECTORSTORE_DIR, SAMPLE_DATA_DIR,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K,
    DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_API_KEY, OPENAI_API_KEY
)
from models.schemas import WorkflowState, LegalResearchAnswer
from rag.document_loader import DocumentLoader
from rag.text_splitter import LegalTextSplitter
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.legal_research_agent import LegalResearchAgent
from utils.helpers import state_to_json, format_file_size

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
    padding: 10px 16px !important;
    text-transform: uppercase !important;
    transition: var(--transition) !important;
    cursor: pointer !important;
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

/* Primary Button */
[data-testid="stFormSubmitButton"] > button,
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue-primary) 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid var(--blue-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 24px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 2px 12px rgba(37,99,235,0.2) !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
.stButton > button[kind="primary"]:hover {
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
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.15;
    color: var(--text-primary);
    margin-bottom: 8px;
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
    border-radius: var(--radius-md);
    padding: 14px 16px;
    cursor: pointer;
    transition: var(--transition);
    height: 100%;
}
.prompt-card:hover {
    border-color: var(--blue-primary);
    background: var(--blue-dim);
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(37,99,235,0.1);
}
.prompt-num {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--blue-primary);
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.prompt-title {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.prompt-desc {
    font-family: var(--font-sans);
    font-size: 0.7rem;
    color: var(--text-muted);
    line-height: 1.4;
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

.pipeline-container {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 20px 0;
    margin: 16px 0;
    overflow-x: auto;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 90px;
    flex-shrink: 0;
}
.pipeline-node.step-1 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 0.05s both; }
.pipeline-node.step-2 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 0.40s both; }
.pipeline-node.step-3 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 0.75s both; }
.pipeline-node.step-4 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 1.10s both; }
.pipeline-node.step-5 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 1.45s both; }
.pipeline-node.step-6 { animation: nodePop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) 1.80s both; }

.pipeline-connector {
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, var(--green), var(--blue-primary));
    min-width: 24px;
    margin-bottom: 28px;
    transform-origin: left center;
}
.pipeline-connector.conn-1 { animation: lineGrow 0.3s ease 0.22s both; }
.pipeline-connector.conn-2 { animation: lineGrow 0.3s ease 0.57s both; }
.pipeline-connector.conn-3 { animation: lineGrow 0.3s ease 0.92s both; }
.pipeline-connector.conn-4 { animation: lineGrow 0.3s ease 1.27s both; }
.pipeline-connector.conn-5 { animation: lineGrow 0.3s ease 1.62s both; }

.pipeline-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--blue-primary);
    margin-bottom: 8px;
    box-shadow: 0 0 8px rgba(37,99,235,0.3);
}
.pipeline-dot.completed {
    background: var(--green);
    box-shadow: 0 0 8px rgba(34,197,94,0.4);
}
.pipeline-node.step-6 .pipeline-dot.completed {
    animation: pulseGlow 2s ease-in-out infinite 2.1s;
}
.pipeline-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
}
.pipeline-value {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-secondary);
    text-align: center;
    margin-top: 2px;
}

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
    padding: 24px;
    margin: 16px 0;
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
    margin-bottom: 16px;
}
.answer-text {
    font-family: var(--font-sans);
    font-size: 0.92rem;
    color: var(--text-primary);
    line-height: 1.7;
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
.finding-row {
    display: flex;
    gap: 14px;
    padding: 12px 0;
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
    st.session_state.llm_provider = DEFAULT_LLM_PROVIDER
if "llm_model" not in st.session_state:
    st.session_state.llm_model = OLLAMA_MODEL
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
            with st.spinner("Loading..."):
                sample_files = list(SAMPLE_DATA_DIR.glob("*.txt"))
                splitter = LegalTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
                total_chunks = 0
                for sf in sample_files:
                    pages_data = DocumentLoader.load_file(sf, sf.name)
                    chunks = splitter.split_pages(pages_data)
                    st.session_state.vector_store.add_chunks(chunks, embedding_service)
                    total_chunks += len(chunks)
                st.success(f"Loaded {len(sample_files)} samples, {total_chunks} chunks")
                st.rerun()

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
                st.caption(f"✓ {ollama_msg}")
            else:
                st.caption(f"⚠️ {ollama_msg}")
    elif provider_choice == "gemini":
        model_choice = st.text_input("Model", value="gemini-2.5-flash", label_visibility="collapsed")
        api_key_input = st.text_input("API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed", placeholder="API Key...")
        is_alive = True
        ollama_msg = ""
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
    <div class="hero-title">Crime Scene<br><span class="highlight">Investigation</span> RAG</div>
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
tab_research, tab_history, tab_documents = st.tabs(["RESEARCH", "HISTORY", "CASE FILES"])

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
        st.session_state.preset_query = ""
        if not user_query or not user_query.strip():
            st.warning("Please enter a valid research question.")
        elif stats['total_chunks'] == 0:
            st.warning("Knowledge base is empty. Upload documents or load samples from the sidebar.")
        else:
            # Live Animated Processing State
            with st.status("AGENT WORKFLOW IN PROGRESS...", expanded=True) as status_container:
                st.write("● Step 1/5: Initializing query & validating input parameters...")
                time.sleep(0.25)
                st.write("● Step 2/5: Identifying research intent & constructing execution plan...")
                time.sleep(0.25)
                st.write("● Step 3/5: Querying vector store & retrieving top evidence chunks...")
                
                state: WorkflowState = agent.run_workflow(query=user_query, top_k=top_k_val)
                
                st.write("● Step 4/5: Executing LLM reasoning & Pydantic schema validation...")
                time.sleep(0.25)
                st.write("● Step 5/5: Synchronizing verifiable source citations & page numbers...")
                time.sleep(0.25)
                status_container.update(label="✓ Analysis Complete — Pipeline Synchronized", state="complete", expanded=False)

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
            st.warning(f"Structured output validation: {state.validation_error}")

        # Workflow Pipeline Visualization (Sequential Load Animation)
        st.markdown('<div class="section-label-blue">Agent Workflow Pipeline</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pipeline-container">
            <div class="pipeline-node step-1">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">Query</div>
                <div class="pipeline-value">Received</div>
            </div>
            <div class="pipeline-connector conn-1"></div>
            <div class="pipeline-node step-2">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">Intent</div>
                <div class="pipeline-value">{intent_val.replace('_', ' ').title()}</div>
            </div>
            <div class="pipeline-connector conn-2"></div>
            <div class="pipeline-node step-3">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">Retrieval</div>
                <div class="pipeline-value">{len(state.sources)} sources</div>
            </div>
            <div class="pipeline-connector conn-3"></div>
            <div class="pipeline-node step-4">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">Evidence</div>
                <div class="pipeline-value">{strength}</div>
            </div>
            <div class="pipeline-connector conn-4"></div>
            <div class="pipeline-node step-5">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">LLM</div>
                <div class="pipeline-value">{provider_choice.title()}</div>
            </div>
            <div class="pipeline-connector conn-5"></div>
            <div class="pipeline-node step-6">
                <div class="pipeline-dot completed"></div>
                <div class="pipeline-label">Answer</div>
                <div class="pipeline-value">{"Validated" if ans else "Raw"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Meta pills
        st.markdown(f"""
        <div class="meta-pills">
            <span class="pill pill-intent">Intent: {intent_val.replace('_', ' ')}</span>
            <span class="pill {strength_class}">Evidence: {strength}</span>
            <span class="pill pill-sources">Sources: {len(state.sources)}</span>
        </div>
        """, unsafe_allow_html=True)

        # ─── ANSWER CARD ─────────────────────
        if ans:
            st.markdown(f"""
            <div class="answer-surface">
                <div class="result-header">
                    <div>
                        <div class="answer-label">AI Research Response</div>
                        <div style="font-size:1.05rem;font-weight:600;color:var(--text-primary);margin-top:4px;">{state.query}</div>
                    </div>
                    <div class="answer-grounded">
                        <span style="width:5px;height:5px;border-radius:50%;background:var(--green);display:inline-block;"></span>
                        Source Grounded
                    </div>
                </div>
                <div class="answer-text">{ans.answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # Key Findings + Evidence in columns
            col_findings, col_evidence = st.columns(2)

            with col_findings:
                st.markdown('<div class="section-label-blue">Key Legal Findings</div>', unsafe_allow_html=True)
                findings_html = ""
                for idx, f in enumerate(ans.key_findings, 1):
                    findings_html += f"""
                    <div class="finding-row">
                        <div class="finding-num">{idx:02d}</div>
                        <div class="finding-text">{f}</div>
                    </div>"""
                st.markdown(findings_html, unsafe_allow_html=True)

            with col_evidence:
                st.markdown('<div class="section-label-blue">Evidence & Witness Summary</div>', unsafe_allow_html=True)
                evidence_html = ""
                for idx, e in enumerate(ans.evidence_summary, 1):
                    evidence_html += f"""
                    <div class="finding-row">
                        <div class="finding-num">{idx:02d}</div>
                        <div class="finding-text">{e}</div>
                    </div>"""
                st.markdown(evidence_html, unsafe_allow_html=True)

            # Limitations
            if ans.limitations:
                st.markdown('<div class="section-label">Limitations & Unknowns</div>', unsafe_allow_html=True)
                lim_html = ""
                for idx, lim in enumerate(ans.limitations, 1):
                    lim_html += f"""
                    <div class="finding-row">
                        <div class="finding-num" style="color:var(--text-muted);">{idx:02d}</div>
                        <div class="finding-text" style="color:var(--text-secondary);">{lim}</div>
                    </div>"""
                st.markdown(lim_html, unsafe_allow_html=True)

        else:
            # Fallback for non-structured answer
            st.markdown(f"""
            <div class="answer-surface">
                <div class="answer-label">Research Response</div>
                <div style="margin-top:4px;font-size:1rem;font-weight:600;color:var(--text-primary);">{state.query}</div>
                <div class="answer-text" style="margin-top:16px;">{state.final_answer}</div>
            </div>
            """, unsafe_allow_html=True)

        # ─── SOURCES ─────────────────────────
        with st.expander(f"EVIDENCE SOURCES  [{len(state.sources)} retrieved]"):
            if not state.sources:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-title">No Sources Retrieved</div>
                    <div class="empty-desc">The query did not match any documents in the knowledge base.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for src in state.sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-top">
                            <span class="source-id">Source {src['citation_num']:02d}</span>
                            <span class="source-page">Page {src['page_number']}</span>
                        </div>
                        <div class="source-name">{src['document_name']}</div>
                        <div class="source-excerpt">"{src['excerpt']}"</div>
                        <div class="source-meta">
                            <span class="source-meta-item">Chunk #{src['chunk_id']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ─── WORKFLOW LOG ─────────────────────
        with st.expander("AGENT WORKFLOW LOG"):
            st.markdown('<div class="section-label">Research Plan</div>', unsafe_allow_html=True)
            for p_idx, plan_item in enumerate(state.plan, 1):
                st.markdown(f"""
                <div class="wf-step">
                    <span class="wf-num">{p_idx:02d}</span>
                    <span class="wf-detail">{plan_item}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:16px;">Execution Log</div>', unsafe_allow_html=True)
            for step in state.steps:
                st.markdown(f"""
                <div class="wf-step">
                    <span class="wf-num">[+]</span>
                    <span class="wf-name">{step.step_name}</span>
                    <span class="wf-detail"> &mdash; {step.detail}</span>
                </div>
                """, unsafe_allow_html=True)

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
            with st.expander(f"{item['timestamp']}  |  {h_intent.replace('_',' ').title()}  |  {item['query'][:80]}"):
                # Meta
                st.markdown(f"""
                <div class="meta-pills">
                    <span class="pill pill-intent">Intent: {h_intent.replace('_', ' ')}</span>
                    <span class="pill pill-sources">Sources: {len(h_state.sources)}</span>
                </div>
                """, unsafe_allow_html=True)

                if h_ans:
                    # Structured answer
                    st.markdown(f"""
                    <div class="answer-surface" style="margin-top:12px;">
                        <div class="answer-label">Response</div>
                        <div class="answer-text" style="margin-top:8px;">{h_ans.get('answer', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if h_ans.get("key_findings"):
                        st.markdown('<div class="section-label" style="margin-top:16px;">Key Findings</div>', unsafe_allow_html=True)
                        findings_html = ""
                        for f_idx, kf in enumerate(h_ans.get("key_findings"), 1):
                            findings_html += f"""
                            <div class="finding-row">
                                <div class="finding-num">{f_idx:02d}</div>
                                <div class="finding-text">{kf}</div>
                            </div>"""
                        st.markdown(findings_html, unsafe_allow_html=True)

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
                <div class="doc-name">{doc_name}</div>
                <div class="doc-status">Ready</div>
            </div>
            """, unsafe_allow_html=True)
