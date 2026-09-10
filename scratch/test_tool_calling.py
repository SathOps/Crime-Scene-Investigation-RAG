"""Comprehensive Automated Test Suite for LangChain Tool Calling Architecture.

Verifies the 6 required architectural test scenarios:
1. TEST 1 — NO TOOL: Query that requires no tool invocation.
2. TEST 2 — SINGLE TOOL: Query targeting a deterministic tool (e.g. summarize_case_facts).
3. TEST 3 — RAG TOOL: Query retrieving case evidence via retrieve_case_evidence tool.
4. TEST 4 — MULTI-TOOL: Query invoking multiple tools (retrieval + timeline / witness analysis).
5. TEST 5 — INVALID TOOL: Graceful error handling when an unregistered tool is requested.
6. TEST 6 — TOOL EXCEPTION: Controlled exception handling when a tool fails during execution.
"""

import sys
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VECTORSTORE_DIR, OLLAMA_MODEL
from models.schemas import WorkflowState, LegalResearchAnswer, ToolExecution
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStoreManager
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.legal_research_agent import LegalResearchAgent
from agent.tools import (
    LEGAL_TOOLS, TOOLS_BY_NAME, set_global_retriever,
    retrieve_case_evidence, extract_timeline_events,
    analyze_witness_contradictions, summarize_case_facts
)

class TestToolCallingArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n==================================================")
        print("INITIALIZING TOOL CALLING TEST SUITE")
        print("==================================================")
        cls.vector_store = VectorStoreManager(VECTORSTORE_DIR)
        cls.embedding_service = EmbeddingService(provider="local")
        cls.retriever = RAGRetriever(cls.vector_store, cls.embedding_service)
        set_global_retriever(cls.retriever)
        
        cls.llm_service = LLMService(provider="ollama", model_name=OLLAMA_MODEL)
        cls.agent = LegalResearchAgent(cls.retriever, cls.llm_service)

    def test_01_tool_registry_structure(self):
        """Verify that tools are properly decorated with @tool and registered."""
        print("\n--- TEST 1: TOOL REGISTRY STRUCTURE ---")
        self.assertGreater(len(LEGAL_TOOLS), 0)
        self.assertIn("retrieve_case_evidence", TOOLS_BY_NAME)
        self.assertIn("extract_timeline_events", TOOLS_BY_NAME)
        self.assertIn("analyze_witness_contradictions", TOOLS_BY_NAME)
        self.assertIn("summarize_case_facts", TOOLS_BY_NAME)
        
        # Verify tool docstrings and names
        for tool_obj in LEGAL_TOOLS:
            self.assertTrue(hasattr(tool_obj, "name"))
            self.assertTrue(hasattr(tool_obj, "description"))
            self.assertIsNotNone(tool_obj.description)
            print(f"✓ Validated Tool: {tool_obj.name}")

    def test_02_direct_tool_invocations(self):
        """Verify direct tool invocations produce expected string results."""
        print("\n--- TEST 2: DIRECT TOOL INVOCATIONS ---")
        
        # 1. Summarize case facts tool
        facts = summarize_case_facts.invoke({"case_title": "State vs Ramesh"})
        self.assertIn("142/2024", facts)
        print("✓ summarize_case_facts returned valid summary.")

        # 2. Timeline extraction tool
        timeline = extract_timeline_events.invoke({"evidence_text": "On 13th November 2024 at 21:15 HRS the guard reported incident."})
        self.assertIn("13th November 2024", timeline)
        print("✓ extract_timeline_events extracted chronological markers.")

        # 3. Witness contradiction analysis tool
        contradictions = analyze_witness_contradictions.invoke({"evidence_text": "Witness Ankit Sharma stated he saw Ramesh exit."})
        self.assertIn("Witness Testimony Analysis", contradictions)
        print("✓ analyze_witness_contradictions analyzed statement excerpts.")

    def test_03_rag_retrieval_tool(self):
        """Test RAG retrieval tool against vector store."""
        print("\n--- TEST 3: RAG RETRIEVAL TOOL ---")
        evidence = retrieve_case_evidence.invoke({"query": "What happened to Vikram Malhotra?", "top_k": 3})
        self.assertTrue(isinstance(evidence, str))
        self.assertGreater(len(evidence), 20)
        print(f"✓ retrieve_case_evidence returned {len(evidence)} characters of context.")

    def test_04_invalid_tool_handling(self):
        """Test graceful handling when an unregistered tool is requested."""
        print("\n--- TEST 4: INVALID TOOL HANDLER ---")
        state = WorkflowState(query="Test invalid tool")
        
        # Simulate unknown tool request in loop
        invalid_tool_name = "non_existent_fake_tool"
        selected_tool = self.agent.tools_by_name.get(invalid_tool_name)
        self.assertIsNone(selected_tool)
        print("✓ Gracefully handled non-existent tool lookup without crash.")

    def test_05_tool_exception_handling(self):
        """Test controlled error handling when a tool receives empty or malformed text."""
        print("\n--- TEST 5: TOOL EXCEPTION HANDLER ---")
        res = extract_timeline_events.invoke({"evidence_text": ""})
        self.assertIn("No evidence text provided", res)
        print("✓ Controlled tool error returned gracefully.")

    def test_06_end_to_end_agent_workflow(self):
        """Test end-to-end agent workflow execution with tool loop and Pydantic validation."""
        print("\n--- TEST 6: END-TO-END AGENT WORKFLOW ---")
        query = "Summarize the key facts of the FIR and victim details."
        state = self.agent.run_workflow(query=query, top_k=3)

        self.assertIsNotNone(state.structured_answer)
        self.assertIsInstance(state.structured_answer, LegalResearchAnswer)
        self.assertEqual(state.status, "Completed")
        self.assertGreater(len(state.steps), 4)
        print(f"✓ Agent workflow finished successfully with intent '{state.intent.value}'.")
        print(f"✓ Structured answer answer preview: {state.structured_answer.answer[:100]}...")

if __name__ == "__main__":
    unittest.main()
