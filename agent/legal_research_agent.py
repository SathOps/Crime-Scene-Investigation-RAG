import re
import json
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from models.schemas import IntentType, DocumentChunk, WorkflowStep, WorkflowState, LegalResearchAnswer, ToolExecution
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.tools import LEGAL_TOOLS, TOOLS_BY_NAME, set_global_retriever

logger = logging.getLogger("LegalResearchAgent")

MAX_TOOL_ITERATIONS = 5

class LegalResearchAgent:
    """Central Agent for Legal Case Research featuring LangChain Tool Calling Architecture.

    Executes the AgenticAI workflow:
    User Query -> Router/Intent -> Bound LLM -> Tool Selection -> Tool Execution -> ToolMessage -> Final Structured Answer
    """

    def __init__(self, retriever: RAGRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm_service = llm_service
        set_global_retriever(self.retriever)
        self.tools = LEGAL_TOOLS
        self.tools_by_name = TOOLS_BY_NAME

    def identify_intent(self, query: str) -> IntentType:
        q_lower = query.lower()

        if any(w in q_lower for w in ["timeline", "chronology", "sequence of events", "dates", "when did"]):
            return IntentType.TIMELINE
        elif any(w in q_lower for w in ["contradict", "inconsistent", "discrepancy", "differ", "conflict", "witness statements"]):
            return IntentType.CONTRADICTION_ANALYSIS
        elif any(w in q_lower for w in ["evidence", "connect", "forensic", "weapon", "blood", "fingerprint"]):
            return IntentType.EVIDENCE_ANALYSIS
        elif any(w in q_lower for w in ["accused", "suspect", "defendant"]):
            return IntentType.ACCUSED_INFORMATION
        elif any(w in q_lower for w in ["witness", "statement", "testimony"]):
            return IntentType.WITNESS_ANALYSIS
        elif any(w in q_lower for w in ["section", "ipc", "crpc", "statute", "act", "provision", "law"]):
            return IntentType.LEGAL_PROVISIONS
        elif any(w in q_lower for w in ["prosecution", "charge"]):
            return IntentType.PROSECUTION_ARGUMENT
        elif any(w in q_lower for w in ["defence", "defense", "alibi"]):
            return IntentType.DEFENCE_ARGUMENT
        elif any(w in q_lower for w in ["court", "judge", "ruling", "verdict", "order"]):
            return IntentType.COURT_FINDINGS
        elif any(w in q_lower for w in ["summary", "what happened", "overview", "brief"]):
            return IntentType.CASE_SUMMARY
        elif any(w in q_lower for w in ["compare", "comparison"]):
            return IntentType.CASE_COMPARISON
        elif any(w in q_lower for w in ["fact", "details"]):
            return IntentType.CASE_FACTS
        else:
            return IntentType.GENERAL_LEGAL_RESEARCH

    def create_research_plan(self, intent: IntentType, query: str) -> List[str]:
        return [
            f"Map query to research intent: {intent.value}",
            "Bind LangChain tool definitions to Chat LLM",
            "Send query to LLM and evaluate dynamic tool calls",
            "Execute selected tool actions and capture ToolMessages",
            "Generate grounded structured JSON response",
            "Validate result using Pydantic LegalResearchAnswer schema"
        ]

    def execute_tool_calling_loop(self, query: str, state: WorkflowState) -> str:
        """Executes iterative LLM tool calling using LangChain messages protocol."""
        chat_model = self.llm_service.get_chat_model()
        
        try:
            llm_with_tools = chat_model.bind_tools(self.tools)
        except Exception as e:
            logger.warning(f"Model bind_tools failed ({e}), falling back to direct tool invocation.")
            llm_with_tools = chat_model

        system_msg = SystemMessage(content=(
            "You are an expert Legal Research Agent. Analyze legal case questions, determine if specialized tools "
            "like `retrieve_case_evidence`, `extract_timeline_events`, `analyze_witness_contradictions`, or "
            "`summarize_case_facts` are required, invoke them appropriately, and synthesize a comprehensive answer."
        ))

        messages = [system_msg, HumanMessage(content=query)]
        iteration = 0

        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            logger.info(f"[AGENT] Tool Loop Iteration {iteration}")
            
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"[AGENT LLM ERROR] {e}")
                break

            messages.append(response)

            # Check if LLM requested tool calls
            tool_calls = getattr(response, "tool_calls", [])
            if not tool_calls:
                logger.info("[AGENT] No tool calls requested by LLM. Exiting tool loop.")
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info(f"[AGENT TOOL REQUEST] {tool_name} with args: {tool_args}")
                state.steps.append(WorkflowStep(
                    step_name=f"Tool Action — {tool_name}",
                    status="in_progress",
                    detail=f"Executing tool '{tool_name}' with args {json.dumps(tool_args)}"
                ))

                selected_tool = self.tools_by_name.get(tool_name)
                if selected_tool is None:
                    err_text = f"Error: Tool '{tool_name}' is not registered in system tools."
                    logger.warning(f"[AGENT TOOL ERROR] {err_text}")
                    messages.append(ToolMessage(content=err_text, tool_call_id=tool_id))
                    state.steps[-1].status = "failed"
                    state.steps[-1].detail = err_text
                    continue

                try:
                    tool_result = selected_tool.invoke(tool_args)
                    result_str = str(tool_result)
                    logger.info(f"[AGENT TOOL RESULT] Success ({len(result_str)} chars)")

                    tool_exec = ToolExecution(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status="completed",
                        result_preview=result_str[:200]
                    )
                    state.executed_tools.append(tool_exec)
                    state.steps[-1].status = "completed"
                    state.steps[-1].detail = f"Tool '{tool_name}' returned result ({len(result_str)} chars)"
                    state.steps[-1].tool_execution = tool_exec

                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

                except Exception as ex:
                    err_msg = f"Tool execution failure: {str(ex)}"
                    logger.error(f"[AGENT TOOL EXCEPTION] {err_msg}")
                    tool_exec = ToolExecution(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        status="failed",
                        result_preview=err_msg
                    )
                    state.executed_tools.append(tool_exec)
                    state.steps[-1].status = "failed"
                    state.steps[-1].detail = err_msg
                    state.steps[-1].tool_execution = tool_exec
                    messages.append(ToolMessage(content=err_msg, tool_call_id=tool_id))

        # Extract final answer content
        final_response_str = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                final_response_str = str(m.content)
                break

        return final_response_str

    def run_workflow(self, query: str, top_k: int = 5) -> WorkflowState:
        state = WorkflowState(query=query)

        # 1. RECEIVE QUERY
        state.steps.append(WorkflowStep("Step 1 — Query Received", "completed", f"Query: '{query}'"))

        # 2. IDENTIFY INTENT
        state.intent = self.identify_intent(query)
        state.steps.append(WorkflowStep("Step 2 — Intent Identified", "completed", f"Intent: {state.intent.value}"))

        # 3. PLAN RESEARCH
        state.plan = self.create_research_plan(state.intent, query)
        state.steps.append(WorkflowStep("Step 3 — Research Plan Created", "completed", f"{len(state.plan)} action steps defined"))

        # 4. RAG EVIDENCE RETRIEVAL (Ensure base chunks are loaded)
        state.steps.append(WorkflowStep("Step 4 — Vector Search & RAG Evidence", "in_progress", f"Querying vector store (Top K = {top_k})"))
        chunks, strength = self.retriever.retrieve_evidence(query, top_k=top_k)
        state.retrieved_chunks = chunks
        state.evidence_strength = strength
        state.steps[-1].status = "completed"
        state.steps[-1].detail = f"Retrieved {len(chunks)} chunks (Evidence Strength: {strength})"

        # 5. EXECUTE LANGCHAIN TOOL CALLING LOOP
        state.steps.append(WorkflowStep("Step 5 — LangChain Tool Calling Loop", "in_progress", "Binding tools to Chat LLM & evaluating tool calls"))
        tool_loop_output = self.execute_tool_calling_loop(query, state)
        state.steps[-1].status = "completed"
        state.steps[-1].detail = f"Tool calling loop finished with {len(state.executed_tools)} tools executed"

        # 6. STRUCTURED RESPONSE & PYDANTIC VALIDATION
        state.steps.append(WorkflowStep("Step 6 — LLM Response Synthesis & Pydantic Validation", "in_progress", "Generating and validating final response"))
        
        answer_obj, raw_resp, err_msg = self.llm_service.generate_structured_response(
            query=query,
            intent=state.intent,
            retrieved_chunks=chunks,
            evidence_strength=strength
        )

        if tool_loop_output and tool_loop_output not in answer_obj.answer:
            answer_obj.answer += f"\n\n### Agent Tool Observations:\n{tool_loop_output}"

        state.structured_answer = answer_obj
        state.raw_model_response = raw_resp
        state.validation_error = err_msg
        state.response = answer_obj.answer
        state.steps[-1].status = "completed"
        state.steps[-1].detail = "Response validated via Pydantic model" if not err_msg else f"Validated with fallback ({err_msg})"

        # 7. SOURCES & CITATIONS
        state.sources = []
        for idx, src in enumerate(answer_obj.sources, 1):
            state.sources.append({
                "citation_num": idx,
                "document_name": src.document_name,
                "document_type": "Legal Document",
                "page_number": src.page_number,
                "chunk_id": src.chunk_id,
                "score": 0.85,
                "excerpt": src.excerpt
            })

        state.steps.append(WorkflowStep("Step 7 — Sources & Citations", "completed", f"{len(state.sources)} verifiable source references attached"))
        state.status = "Completed"

        return state

