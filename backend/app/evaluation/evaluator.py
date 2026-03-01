# backend/app/evaluation/evaluator.py

from backend.app.agent.state import AgentState
from backend.app.evaluation.citation_check import check_citation
from backend.app.evaluation.faithfulness import check_faithfulness


def evaluate(state: AgentState) -> dict:
    citation_result = check_citation(state)
    faithfulness_result = check_faithfulness(state)

    citation_valid = citation_result["citation_valid"]
    faithful = faithfulness_result["faithful"]

    reliability_flag = citation_valid and faithful

    return {
        "citation_valid": citation_valid,
        "faithful": faithful,
        "reliability_flag": reliability_flag,
        "debug_info": {
            "citation_reason": citation_result["citation_reason"],
            "faithfulness_reason": faithfulness_result["faithfulness_reason"],
        }
    }