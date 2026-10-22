"""Agent package exports.

Use a lazy export so importing a lightweight specialist module for unit tests does not
eagerly import the entire retrieval and UI dependency graph.
"""

__all__ = ["LegalResearchAgent"]


def __getattr__(name):
    if name == "LegalResearchAgent":
        from .legal_research_agent import LegalResearchAgent
        return LegalResearchAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
