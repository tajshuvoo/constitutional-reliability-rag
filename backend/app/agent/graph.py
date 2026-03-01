from langgraph.graph import StateGraph, START, END

from backend.app.agent.state import AgentState
from backend.app.agent.retrieve_node import retrieve_node
from backend.app.agent.generate_node import generate_node
from backend.app.agent.evaluate_node import evaluate_node
from backend.app.agent.correction_node import correction_node


# Maximum allowed correction attempts
MAX_CORRECTIONS = 1


def route_after_evaluation(state: AgentState):
    """
    Routing logic after evaluation step.

    If answer is reliable -> END.
    If unreliable and correction attempts < limit -> correction node.
    If correction attempts exceeded -> END (fail safe).
    """

    # Case 1: Reliable answer
    if state.reliability_flag:
        return END

    # Case 2: Retry limit reached
    if state.correction_attempts >= MAX_CORRECTIONS:
        return END

    # Case 3: Try correction
    return "correction"


def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("correction", correction_node)

    # Main flow
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")

    # Conditional routing
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            END: END,
            "correction": "correction",
        },
    )

    # After correction, re-evaluate
    g.add_edge("correction", "evaluate")

    return g.compile()