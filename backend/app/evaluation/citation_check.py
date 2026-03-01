# backend/app/evaluation/citation_check.py

import re
from typing import Set, Dict
from backend.app.agent.state import AgentState


ARTICLE_PATTERN = re.compile(r"\barticle\s+(\d+)\b", re.IGNORECASE)
NOT_FOUND_PHRASE = "the answer is not found in the retrieved constitutional articles."


def extract_cited_articles(answer: str) -> Set[str]:
    return set(ARTICLE_PATTERN.findall(answer))


def check_citation(state: AgentState) -> Dict:
    raw_answer = state.draft_answer or ""
    answer_clean = raw_answer.strip()
    answer_lower = answer_clean.lower()

    # -------------------------
    # STRICT NOT-FOUND POLICY
    # -------------------------

    if answer_lower.startswith(NOT_FOUND_PHRASE):

        # If exactly equal → valid refusal
        if answer_lower == NOT_FOUND_PHRASE:
            return {
                "citation_valid": True,
                "citation_reason": "Strict valid not-found response."
            }

        # If additional text exists → invalid
        return {
            "citation_valid": False,
            "citation_reason": "Not-found response contains additional explanation."
        }

    # -------------------------
    # Normal citation logic
    # -------------------------

    cited_articles = extract_cited_articles(raw_answer)

    retrieved_numbers = {
        str(article.section_no_en)
        for article in state.retrieved_articles
        if article.section_no_en is not None
    }

    if not cited_articles:
        return {
            "citation_valid": False,
            "citation_reason": "No article citation found."
        }

    if not cited_articles.intersection(retrieved_numbers):
        return {
            "citation_valid": False,
            "citation_reason": "Cited article not retrieved."
        }

    return {
        "citation_valid": True,
        "citation_reason": "Valid grounded citation."
    }