# Multi-Agent RAG Migration Plan

This branch is reserved for converting the existing single-agent Crime Scene Investigation RAG application into a multi-agent system while preserving the working `main` branch.

## Baseline

The existing application includes:
- Streamlit UI in `app.py`
- `LegalResearchAgent` in `agent/legal_research_agent.py`
- Hybrid evidence retrieval in `rag/retriever.py`
- LLM provider handling in `llm/llm_service.py`
- Structured response schemas in `models/schemas.py`
- Query normalization and intent classification in `utils/`
- Existing query-understanding evaluation and pipeline verification scripts

## Proposed target architecture

```text
Streamlit UI
    |
Research Coordinator (workflow owner)
    |-- Query/Intent Router
    |-- Evidence Retrieval Agent (reuse RAGRetriever)
    |-- Timeline Analysis Agent (when relevant)
    |-- Witness / Contradiction Analysis Agent (when relevant)
    |-- Case Facts Agent (when relevant)
    |-- Synthesis Agent (uses evidence and delegated findings)
    |-- Citation / Grounding Validator
    |
LegalResearchAnswer (existing Pydantic contract)
    |
Streamlit result UI
```

The coordinator should route only the necessary specialist(s) for a query. The retrieval layer, vector store, provider configuration, UI contract, and Pydantic output schema should remain reusable. Agents should receive evidence explicitly rather than relying on shared mutable global context wherever practical.

## Incremental implementation and commit plan

1. **Baseline and branch setup** — keep `main` unchanged; run current checks locally and record baseline behavior.
2. **Agent contracts** — add typed request/result objects for specialist agents and coordinator handoffs; tests for contracts.
3. **Specialist agents** — implement focused timeline, witness/contradiction, and case-facts analysis on supplied evidence; tests with deterministic inputs.
4. **Coordinator** — route based on existing intent classification, retrieve evidence once where possible, call applicable specialists, and track steps in `WorkflowState`.
5. **Evidence-grounded synthesis** — combine specialist results with the existing structured response contract; retain source references and limitations.
6. **Streamlit integration** — expose agent activity and maintain existing UI behavior/session history.
7. **Regression and documentation** — run existing tests plus multi-agent tests, update README, and document setup and limitations.

Each step should be a separate, focused commit. Keep changes on `feature/multi-agent-rag` until local tests pass. Do not merge into `main` until reviewed.

## Local Antigravity workflow (macOS)

Open a terminal and run:

```bash
cd /path/to/your/projects
git clone https://github.com/SathOps/Crime-Scene-Investigation-RAG.git
cd Crime-Scene-Investigation-RAG
git switch feature/multi-agent-rag
```

If the repository is already on your Mac, use its project directory instead and run:

```bash
git status
git fetch origin
git switch feature/multi-agent-rag
git pull --ff-only origin feature/multi-agent-rag
```

Before switching branches, commit or stash any local uncommitted work. Then open this repository folder in Antigravity.

Create/activate an environment and install dependencies if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your own provider settings to `.env`; never commit `.env` or API keys. For the local Ollama provider, ensure Ollama is installed and the model configured in `.env` has been pulled. Launch with:

```bash
streamlit run app.py
```

## Commit and push pattern

After each verified change:

```bash
git status
git diff
# run the relevant test(s)
git add <specific-files>
git commit -m "feat: <small focused change>"
git push -u origin feature/multi-agent-rag
```

For subsequent commits, `git push origin feature/multi-agent-rag` is sufficient. Avoid `git add .` until you have inspected `git status`, so local data, environment files, or generated vector stores do not get committed.

## Safety and compatibility rules

- Preserve `main` and its existing single-agent behavior as a rollback point.
- Do not modify unrelated UI styling, retrieval scoring, dependency versions, or prompt behavior as part of the agent refactor unless tests show it is necessary.
- Keep the current `LegalResearchAnswer` schema compatible with the Streamlit rendering path.
- Do not claim that model output is verified legal advice; return source-backed findings and explicit knowledge-base limitations.
- Add tests alongside each feature, not only at the end.
- Check `git diff` before every commit and run relevant existing tests before pushing.
