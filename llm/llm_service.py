import json
import re
import requests
from typing import List, Dict, Any, Tuple, Optional
from pydantic import ValidationError

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from models.schemas import DocumentChunk, IntentType, LegalResearchAnswer, SourceModel
from prompts.legal_research_prompt import LEGAL_RESEARCH_PROMPT_TEMPLATE

class LLMService:
    """LangChain-powered LLM Service supporting Ollama (ChatOllama), Gemini, OpenAI,
    and a deterministic Local Fallback generator with Pydantic Structured Output Validation.
    """

    def __init__(self, provider: str = "ollama", model_name: str = OLLAMA_MODEL, api_key: str = "", base_url: str = OLLAMA_BASE_URL):
        self.provider = provider.lower()
        self.model_name = model_name or OLLAMA_MODEL
        self.api_key = api_key
        self.base_url = base_url or OLLAMA_BASE_URL

    def get_chat_model(self):
        """Returns a LangChain compatible Chat model instance based on active provider configuration."""
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.1
            )
        elif self.provider == "openai" and self.api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model_name or "gpt-4o",
                api_key=self.api_key,
                temperature=0.1
            )
        else:
            # Default to Ollama fallback instance
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.1
            )

    @staticmethod
    def check_ollama_status(target_model: str = OLLAMA_MODEL) -> Tuple[bool, str]:
        """Check if local Ollama server is running and accessible and if target model exists."""
        try:
            res = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2)
            if res.status_code == 200:
                models_info = res.json()
                models_list = [m.get("name") for m in models_info.get("models", [])]
                if not models_list:
                    return False, f"Ollama active, but 0 models pulled. Run 'ollama pull {target_model}' in terminal or switch Provider to 'local'."
                
                has_target = any(target_model in m or m.startswith(target_model.split(':')[0]) for m in models_list)
                if not has_target:
                    return True, f"Ollama active ({', '.join(models_list)}). Model '{target_model}' not found — run 'ollama pull {target_model}'."
                return True, f"Ollama operational ({', '.join(models_list)})"
            return False, f"Ollama returned HTTP status {res.status_code}"
        except Exception as e:
            return False, f"Unable to connect to Ollama at {OLLAMA_BASE_URL}. Ensure Ollama service is running."

    def generate_structured_response(
        self,
        query: str,
        intent: IntentType,
        retrieved_chunks: List[DocumentChunk],
        evidence_strength: str
    ) -> Tuple[LegalResearchAnswer, str, str]:
        """Generates a structured legal research response.

        Returns: (LegalResearchAnswer, raw_response_str, error_message)
        """

        sources_list = [
            SourceModel(
                document_name=c.document_name,
                page_number=str(c.page_number),
                chunk_id=c.chunk_id,
                excerpt=c.text[:250].replace('\n', ' ')
            )
            for c in retrieved_chunks
        ]

        intent_str = intent.value if isinstance(intent, IntentType) else str(intent)

        if not retrieved_chunks or evidence_strength == "Insufficient":
            answer_obj = LegalResearchAnswer(
                query=query,
                intent=intent_str,
                answer="I could not find sufficient supporting information in the provided case documents.",
                key_findings=["No supporting facts found in the current knowledge base."],
                evidence_summary=["No relevant document passages matched the query parameters."],
                limitations=[f"Query '{query}' cannot be resolved with available case records."],
                sources=[]
            )
            return answer_obj, json.dumps(answer_obj.model_dump(), indent=2), ""

        formatted_context = self._format_chunks(retrieved_chunks)

        # ─────────────────────────────────────────────
        # PROVIDER 1: OLLAMA (LANGCHAIN ChatOllama)
        # ─────────────────────────────────────────────
        if self.provider == "ollama":
            is_alive, msg = LLMService.check_ollama_status(self.model_name)
            if not is_alive and "not found" not in msg:
                error_msg = f"Ollama Error: {msg}"
                return self._build_fallback_answer(query, intent_str, retrieved_chunks, error_msg), "", error_msg

            try:
                from langchain_ollama import ChatOllama
                llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=0.1,
                    format="json"
                )
                
                chain = LEGAL_RESEARCH_PROMPT_TEMPLATE | llm
                response = chain.invoke({
                    "question": query,
                    "intent": intent_str,
                    "evidence_strength": evidence_strength,
                    "context": formatted_context
                })
                
                raw_content = response.content if hasattr(response, 'content') else str(response)
                return self._parse_and_validate_json(raw_content, query, intent_str, sources_list, retrieved_chunks)

            except Exception as e:
                error_msg = f"LangChain Ollama invocation failed: {str(e)}"
                return self._build_fallback_answer(query, intent_str, retrieved_chunks, error_msg), "", error_msg

        # ─────────────────────────────────────────────
        # PROVIDER 2: GEMINI
        # ─────────────────────────────────────────────
        elif self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt_messages = LEGAL_RESEARCH_PROMPT_TEMPLATE.format_messages(
                    question=query,
                    intent=intent_str,
                    evidence_strength=evidence_strength,
                    context=formatted_context
                )
                prompt_text = "\n\n".join([m.content for m in prompt_messages])

                res = client.models.generate_content(
                    model=self.model_name or "gemini-2.5-flash",
                    contents=prompt_text
                )
                raw_content = res.text.strip() if res and res.text else ""
                return self._parse_and_validate_json(raw_content, query, intent_str, sources_list, retrieved_chunks)
            except Exception as e:
                error_msg = f"Gemini API invocation failed: {str(e)}"
                return self._build_fallback_answer(query, intent_str, retrieved_chunks, error_msg), "", error_msg

        # ─────────────────────────────────────────────
        # PROVIDER 3: OPENAI
        # ─────────────────────────────────────────────
        elif self.provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                prompt_messages = LEGAL_RESEARCH_PROMPT_TEMPLATE.format_messages(
                    question=query,
                    intent=intent_str,
                    evidence_strength=evidence_strength,
                    context=formatted_context
                )
                messages_payload = [{"role": ("system" if m.type == "system" else "user"), "content": m.content} for m in prompt_messages]

                res = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=messages_payload,
                    temperature=0
                )
                raw_content = res.choices[0].message.content.strip()
                return self._parse_and_validate_json(raw_content, query, intent_str, sources_list, retrieved_chunks)
            except Exception as e:
                error_msg = f"OpenAI API invocation failed: {str(e)}"
                return self._build_fallback_answer(query, intent_str, retrieved_chunks, error_msg), "", error_msg

        # ─────────────────────────────────────────────
        # FALLBACK: LOCAL DETERMINISTIC GENERATOR
        # ─────────────────────────────────────────────
        fallback_obj = self._build_fallback_answer(query, intent_str, retrieved_chunks, "Using zero-config local fallback")
        return fallback_obj, json.dumps(fallback_obj.model_dump(), indent=2), ""

    def _format_chunks(self, chunks: List[DocumentChunk]) -> str:
        blocks = []
        for idx, c in enumerate(chunks, 1):
            blocks.append(
                f"[{idx}] Document: {c.document_name} | Page: {c.page_number} | Chunk: {c.chunk_id} | Type: {c.document_type}\n"
                f"Content: {c.text}"
            )
        return "\n\n".join(blocks)

    def _clean_and_extract_json(self, raw_text: str) -> str:
        """Extracts JSON substring and fixes common formatting issues."""
        text = raw_text.strip()
        
        # 1. Strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        # 2. Extract substring between first '{' and last '}'
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx+1]
        elif start_idx != -1:
            text = text[start_idx:]

        # 3. Clean trailing commas before closing braces/brackets
        text = re.sub(r',\s*([\}\]])', r'\1', text)

        # 4. Attempt auto-closure if truncated
        if not text.endswith("}"):
            if text.count('"') % 2 != 0:
                text += '"'
            open_brackets = text.count("[") - text.count("]")
            if open_brackets > 0:
                text += "]" * open_brackets
            open_braces = text.count("{") - text.count("}")
            if open_braces > 0:
                text += "}" * open_braces

        return text

    def _parse_and_validate_json(
        self,
        raw_content: str,
        query: str,
        intent: str,
        sources_list: List[SourceModel],
        chunks: List[DocumentChunk]
    ) -> Tuple[LegalResearchAnswer, str, str]:
        """Extracts, repairs, and validates JSON against LegalResearchAnswer Pydantic model (Experiment 4)."""
        json_str = self._clean_and_extract_json(raw_content)

        data = None
        try:
            data = json.loads(json_str)
        except Exception:
            data = self._regex_extract_fields(raw_content, query, intent)

        if isinstance(data, dict):
            # Ensure required schema keys exist
            data["query"] = data.get("query") or query
            data["intent"] = data.get("intent") or intent
            data["answer"] = data.get("answer") or self._synthesize_answer_from_chunks(chunks)
            
            findings = data.get("key_findings")
            if not isinstance(findings, list) or not findings:
                data["key_findings"] = self._extract_key_findings_from_chunks(chunks)
            else:
                data["key_findings"] = [str(f) for f in findings if f]

            ev_summary = data.get("evidence_summary")
            if not isinstance(ev_summary, list) or not ev_summary:
                data["evidence_summary"] = [f"Synthesized evidence from {len(chunks)} document passage(s)."]
            else:
                data["evidence_summary"] = [str(e) for e in ev_summary if e]

            limits = data.get("limitations")
            if not isinstance(limits, list) or not limits:
                data["limitations"] = ["Information strictly bounded by uploaded case documents."]
            else:
                data["limitations"] = [str(l) for l in limits if l]

            data["sources"] = sources_list

            try:
                structured_obj = LegalResearchAnswer.model_validate(data)
                return structured_obj, raw_content, ""
            except ValidationError:
                pass

        # If all JSON parsing failed, construct rich fallback answer
        fallback_obj = self._build_fallback_answer(query, intent, chunks)
        return fallback_obj, raw_content, ""

    def _regex_extract_fields(self, raw_text: str, query: str, intent: str) -> Dict[str, Any]:
        """Regex helper to rescue fields if JSON is partially malformed."""
        extracted = {"query": query, "intent": intent}
        
        # Try finding "answer": "..."
        ans_match = re.search(r'"answer"\s*:\s*"(.*?)"(?:\s*,\s*"|\s*\})', raw_text, re.DOTALL)
        if ans_match:
            extracted["answer"] = ans_match.group(1).replace('\\"', '"').replace('\\n', '\n')

        # Try finding key_findings
        kf_match = re.search(r'"key_findings"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
        if kf_match:
            findings = re.findall(r'"([^"]+)"', kf_match.group(1))
            if findings:
                extracted["key_findings"] = findings

        return extracted

    def _synthesize_answer_from_chunks(self, chunks: List[DocumentChunk]) -> str:
        paragraphs = []
        for c in chunks[:3]:
            clean = c.text.strip().replace("\n\n", " ")
            paragraphs.append(f"According to {c.document_name} (Page {c.page_number}): {clean}")
        return "\n\n".join(paragraphs) if paragraphs else "Retrieved evidence indexed from case files."

    def _extract_key_findings_from_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        findings = []
        for c in chunks:
            clean = c.text.strip().replace("\n\n", " ")
            sentences = [s.strip() for s in clean.split(".") if len(s.strip()) > 15]
            if sentences:
                findings.append(f"[{c.document_name}]: {sentences[0]}.")
        return findings if findings else ["Case facts extracted from indexed documents."]

    def _build_fallback_answer(self, query: str, intent: str, chunks: List[DocumentChunk], note: str = "") -> LegalResearchAnswer:
        """Constructs a rich, grounded LegalResearchAnswer object directly from retrieved document evidence."""
        findings = self._extract_key_findings_from_chunks(chunks)
        evidence_summaries = [f"Evidence Passage {i:02d} ({c.document_name}, Page {c.page_number}): {c.text[:180]}..." for i, c in enumerate(chunks, 1)]
        full_answer = self._synthesize_answer_from_chunks(chunks)
        sources = [
            SourceModel(
                document_name=c.document_name,
                page_number=str(c.page_number),
                chunk_id=c.chunk_id,
                excerpt=c.text[:250].replace('\n', ' ')
            )
            for c in chunks
        ]

        return LegalResearchAnswer(
            query=query,
            intent=intent,
            answer=full_answer,
            key_findings=findings,
            evidence_summary=evidence_summaries,
            limitations=["Information strictly bounded by uploaded case documents."],
            sources=sources
        )
