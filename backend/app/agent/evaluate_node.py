import re
from typing import Set

from backend.app.agent.state import AgentState


ARTICLE_PATTERN = re.compile(r"\barticle\s+(\d+)\b", re.IGNORECASE)


def extract_cited_articles(answer: str) -> Set[str]:
    """
    Extract article numbers cited in answer.
    Matches:
    - Article 143
    - article 143
    """
    matches = ARTICLE_PATTERN.findall(answer)
    return set(matches)


def evaluate_node(state: AgentState) -> AgentState:
    answer = state.draft_answer or ""

    # Extract cited article numbers from answer
    cited_articles = extract_cited_articles(answer)

    # Extract retrieved article numbers
    retrieved_numbers = {
        str(article.section_no_en)
        for article in state.retrieved_articles
        if article.section_no_en is not None
    }

    # Case 1: No citation found
    if not cited_articles:
        state.citation_valid = False
        state.reliability_flag = False
        state.debug_info["reason"] = "No article citation found in answer."
        state.final_answer = state.draft_answer
        return state

    # Case 2: Citation exists but not among retrieved
    if not cited_articles.intersection(retrieved_numbers):
        state.citation_valid = False
        state.reliability_flag = False
        state.debug_info["reason"] = "Cited article not in retrieved set."
        state.final_answer = state.draft_answer
        return state

    # Passed
    state.citation_valid = True
    state.reliability_flag = True
    state.final_answer = state.draft_answer

    return state