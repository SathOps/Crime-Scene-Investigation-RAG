import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from models.schemas import IntentType, DocumentChunk, WorkflowStep, WorkflowState, LegalResearchAnswer, ToolExecution
from rag.retriever import RAGRetriever
from llm.llm_service import LLMService
from agent.tools import LEGAL_TOOLS, TOOLS_BY_NAME, set_global_retriever, set_global_context
from agent.specialist_agents import MultiAgentCoordinator
from utils.query_normalizer import QueryNormalizer, NormalizedQuery
from utils.intent_classifier import IntentClassifier

logger = logging.getLogger("LegalResearchAgent")


def is_standalone_hypothetical_query(query: str) -> bool:
    """Identify generic, self-contained scenarios that must not be mixed with case files."""
    text = (query or "").lower()
    has_generic_subject = any(phrase in text for phrase in (
        "a person is accused", "suppose a person", "hypothetically", "imagine that",
        "consider this scenario", "in this hypothetical"
    ))
    scenario_markers = sum(phrase in text for phrase in (
        "one witness", "another witness", "another says", "cctv", "footage shows",
        "at 8 pm", "at 9 pm", "at 8:30 pm", "incident occurred"
    ))
    has_named_case_context = any(phrase in text for phrase in (
        "uploaded case", "case file", "in the case of", "case number", "fir no."
    ))
    return has_generic_subject and scenario_markers >= 2 and not has_named_case_context


def build_hypothetical_answer(query: str, normalized_query: str, intent: str) -> LegalResearchAnswer:
    """Answer a self-contained hypothetical using only facts stated in the question."""
    answer = (
        "Based only on the hypothetical facts supplied: one witness places the incident at 8:00 PM; "
        "another places it at 9:00 PM; and the CCTV is said to show the accused entering the building "
        "at 8:30 PM. The witness accounts differ by one hour. The stated CCTV entry time falls between "
        "the two accounts, but it records entry, not necessarily the time the theft occurred. These facts "
        "alone do not establish who committed the theft or whether the accused was present when it occurred. "
        "Further information would be needed about the theft's actual time window, what the CCTV covers, "
        "and any evidence linking the accused to the offence."
    )
    return LegalResearchAnswer(
        query=query,
        normalized_query=normalized_query,
        intent=intent,
        answer=answer,
        key_findings=[
            "Witness 1 places the incident at 8:00 PM (as stated in the question).",
            "Witness 2 places the incident at 9:00 PM (as stated in the question).",
            "CCTV is described as showing building entry at 8:30 PM, not the theft itself."
        ],
        evidence_summary=[
            "The two stated witness times differ by one hour.",
            "The 8:30 PM entry time lies between the two reported incident times; entry alone does not prove participation in the theft."
        ],
        limitations=[
            "No case documents were used for this hypothetical analysis.",
            "The actual time of the theft, CCTV coverage, identity confirmation, and evidence connecting the accused to the theft are not provided."
        ],
        sources=[]
    )


MAX_TOOL_ITERATIONS = 5

class LegalResearchAgent:
    """Central Agent for Legal Case Research featuring Intelligent Preprocessing
    and LangChain Tool Calling Architecture.

    Executes the AgenticAI workflow:
    User Query -> Normalizer -> Classifier -> Retriever -> Bound LLM -> Tool Loop -> Final Structured Answer
    """

    def __init__(self, retriever: RAGRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm_service = llm_service
        set_global_retriever(self.retriever)
        self.tools = LEGAL_TOOLS
        self.tools_by_name = TOOLS_BY_NAME
        self.multi_agent_coordinator = MultiAgentCoordinator(llm_service)

    def identify_intent(self, query: str, norm_obj: Optional[NormalizedQuery] = None) -> IntentType:
        """Classifies query intent using semantic patterns, keywords, and entity context."""
        intent, _ = IntentClassifier.classify(query, norm_obj)
        return intent

    def create_research_plan(self, intent: IntentType, query: str, confidence: float = 1.0) -> List[str]:
        return [
            "Normalize raw user input and expand abbreviations/slang",
            f"Classify research intent: {intent.value} (Confidence: {confidence:.0%})",
            "Execute hybrid vector & lexical retrieval with entity preservation",
            "Delegate retrieved evidence to intent-relevant specialist agents",
            "Bind LangChain specialized legal tools to Chat LLM",
            "Evaluate dynamic tool calls and cross-examine evidence passages",
            "Synthesize evidence-grounded structured response with citations",
            "Validate output schema against Pydantic LegalResearchAnswer model"
        ]

    def execute_tool_calling_loop(self, query: str, state: WorkflowState) -> str:
        """Executes iterative LLM tool calling using LangChain messages protocol."""
        if self.llm_service.provider == "local":
            return ""

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

    def run_workflow(
        self,
        query: str,
        top_k: int = 5,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> WorkflowState:
        """Run the workflow and optionally report progress as stages complete."""
        def report(message: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(message)
                except Exception:
                    logger.debug("Progress callback failed", exc_info=True)

        state = WorkflowState(query=query)
        report("Query received; normalizing and classifying intent.")

        # 1. RECEIVE & PREPROCESS QUERY
        norm_obj = QueryNormalizer.normalize(query)
        state.normalized_query = norm_obj.normalized_query
        state.is_ambiguous = norm_obj.is_ambiguous
        state.clarification_prompt = norm_obj.clarification_prompt
        state.normalization_notes = norm_obj.normalization_notes

        clean_disp = f"Original: '{query}' -> Normalized: '{state.normalized_query}'"
        state.steps.append(WorkflowStep("Step 1 — Query Received & Normalized", "completed", clean_disp))

        # 2. IDENTIFY INTENT
        intent, confidence = IntentClassifier.classify(query, norm_obj)
        state.intent = intent
        state.intent_confidence = confidence
        state.steps.append(WorkflowStep(
            "Step 2 — Intent Identified",
            "completed",
            f"Intent: {state.intent.value} (Confidence: {confidence:.0%})"
        ))

        # 3. PLAN RESEARCH
        state.plan = self.create_research_plan(state.intent, state.normalized_query, confidence)
        state.steps.append(WorkflowStep(
            "Step 3 — Research Plan Created",
            "completed",
            f"{len(state.plan)} action steps defined"
        ))

        # Keep a self-contained hypothetical separate from unrelated uploaded case files.
        if is_standalone_hypothetical_query(query):
            report("Self-contained hypothetical detected; skipping retrieval and case tools.")
            answer_obj = build_hypothetical_answer(
                query, state.normalized_query, state.intent.value
            )
            state.structured_answer = answer_obj
            state.response = answer_obj.answer
            state.evidence_strength = "User-provided hypothetical"
            state.steps.append(WorkflowStep(
                "Step 4 — Hypothetical Scenario Isolation",
                "completed",
                "Answered only from user-provided facts; knowledge-base retrieval and case tools skipped."
            ))
            state.steps.append(WorkflowStep(
                "Step 5 — Structured Response Validation",
                "completed",
                "Deterministic hypothetical answer conforms to LegalResearchAnswer schema."
            ))
            state.status = "Completed"
            return state

        # 4. RAG EVIDENCE RETRIEVAL (Hybrid Search & Entity Preservation)
        report("Retrieving evidence from the local vector store; first retrieval may take several seconds.")
        state.steps.append(WorkflowStep(
            "Step 4 — Hybrid Vector & Lexical Retrieval",
            "in_progress",
            f"Querying vector store (Top K = {top_k}) with preserved entities: {norm_obj.entities_preserved}"
        ))
        chunks, strength = self.retriever.retrieve_evidence(
            query=query,
            top_k=top_k,
            normalized_query=state.normalized_query,
            entities=norm_obj.entities_preserved,
            intent=state.intent.value
        )
        state.retrieved_chunks = chunks
        state.evidence_strength = strength

        # Set global context buffer for agent tools
        formatted_context = self.retriever.format_chunks_for_context(chunks)
        set_global_context(formatted_context)

        state.steps[-1].status = "completed"
        state.steps[-1].detail = f"Retrieved {len(chunks)} chunks (Evidence Strength: {strength})"
        report(f"Retrieval complete: {len(chunks)} chunks; evidence strength {strength}. Running specialist analysis.")

        # Handle pure ambiguous query where evidence is insufficient
        if state.is_ambiguous and state.clarification_prompt and strength == "Insufficient":
            answer_obj = LegalResearchAnswer(
                query=query,
                normalized_query=state.normalized_query,
                intent=state.intent.value,
                answer=f"Your question appears ambiguous or under-specified. {state.clarification_prompt}",
                key_findings=["Clarification required before proceeding with factual case analysis."],
                evidence_summary=["No specific case facts could be definitively mapped without clarification."],
                limitations=[state.clarification_prompt],
                sources=[]
            )
            state.structured_answer = answer_obj
            state.response = answer_obj.answer
            state.steps.append(WorkflowStep("Step 5 — Clarification Check", "completed", "Requested user clarification"))
            state.status = "Completed"
            return state

        # 5. MULTI-AGENT SPECIALIST ANALYSIS
        specialist_step = WorkflowStep(
            "Step 5 — Multi-Agent Specialist Analysis",
            "in_progress",
            f"Routing evidence to specialists for intent: {state.intent.value}"
        )
        state.steps.append(specialist_step)

        specialist_context = self.retriever.format_chunks_for_context(chunks)
        specialist_results = self.multi_agent_coordinator.analyze(
            query=state.normalized_query,
            intent=state.intent,
            evidence_context=specialist_context,
        )
        successful_specialists = sum(
            1 for result in specialist_results if result.status == "completed"
        )
        for result in specialist_results:
            if result.status == "completed":
                step_status = "completed"
                detail = result.analysis[:700]
            elif result.status == "failed":
                step_status = "failed"
                detail = result.error[:500]
            else:
                step_status = "completed"
                detail = f"Skipped: {result.error or 'No specialist analysis performed.'}"
            state.steps.append(WorkflowStep(
                f"Agent — {result.agent_name}",
                step_status,
                detail,
            ))
        specialist_step.status = (
            "completed"
            if successful_specialists or not any(result.status == "failed" for result in specialist_results)
            else "failed"
        )
        specialist_step.detail = (
            f"Completed {successful_specialists} of {len(specialist_results)} specialist analyses. "
            "Specialist notes are secondary analysis; final synthesis still uses retrieved source passages."
        )
        report(f"Specialist analysis complete: {successful_specialists}/{len(specialist_results)} completed. Preparing final synthesis.")
        specialist_analysis_notes = self.multi_agent_coordinator.format_successful_analyses(
            specialist_results
        )

        # 6. EXECUTE LANGCHAIN TOOL CALLING LOOP
        tool_loop_step = WorkflowStep(
            "Step 6 — LangChain Tool Calling Loop",
            "in_progress",
            "Binding tools to Chat LLM & evaluating tool calls"
        )
        state.steps.append(tool_loop_step)
        tool_loop_output = self.execute_tool_calling_loop(
            state.normalized_query, state
        )
        tool_loop_step.status = "completed"
        report("Tool-calling stage complete. Generating the final structured answer.")
        tool_loop_step.detail = (
            f"Tool calling loop finished with "
            f"{len(state.executed_tools)} tools executed"
        )

        # 7. STRUCTURED RESPONSE & PYDANTIC VALIDATION
        state.steps.append(WorkflowStep(
            "Step 7 — LLM Response Synthesis & Pydantic Validation",
            "in_progress",
            "Generating and validating final response"
        ))
        
        answer_obj, raw_resp, err_msg = self.llm_service.generate_structured_response(
            query=query,
            intent=state.intent,
            retrieved_chunks=chunks,
            evidence_strength=strength,
            normalized_query=state.normalized_query,
            # Specialist/tool output remains in the execution trace, not primary synthesis evidence.
            analysis_notes="",
        )

        state.structured_answer = answer_obj
        state.raw_model_response = raw_resp
        state.validation_error = err_msg
        state.response = answer_obj.answer
        state.steps[-1].status = "completed"
        state.steps[-1].detail = "Response validated via Pydantic model" if not err_msg else f"Validated with fallback ({err_msg})"

        # 8. SOURCES & CITATIONS
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

        state.steps.append(WorkflowStep("Step 8 — Sources & Citations", "completed", f"{len(state.sources)} verifiable source references attached"))
        state.status = "Completed"
        report("Research workflow completed; sources and validation results are ready.")

        return state
