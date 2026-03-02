# backend/app/agent/graph.py

from langgraph.graph import StateGraph, START, END

from backend.app.agent.state import AgentState
from backend.app.agent.planner_node import planner_node
from backend.app.agent.retrieve_node import retrieve_node
from backend.app.agent.generate_node import generate_node
from backend.app.agent.evaluate_node import evaluate_node
from backend.app.agent.correction_node import correction_node


MAX_CORRECTIONS = 1


def route_after_evaluation(state: AgentState):

    if state.reliability_flag:
        return END

    if state.correction_attempts >= MAX_CORRECTIONS:
        return END

    return "correction"


def build_graph():

    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("correction", correction_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")

    g.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            END: END,
            "correction": "correction",
        },
    )

    g.add_edge("correction", "evaluate")

    return g.compile()