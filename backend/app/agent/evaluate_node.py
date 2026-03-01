# backend/app/agent/evaluate_node.py

from backend.app.agent.state import AgentState
from backend.app.evaluation.evaluator import evaluate


def evaluate_node(state: AgentState) -> AgentState:
    metrics = evaluate(state)

    state.citation_valid = metrics["citation_valid"]
    state.reliability_flag = metrics["reliability_flag"]
    state.debug_info = metrics["debug_info"]
    state.final_answer = state.draft_answer

    return state