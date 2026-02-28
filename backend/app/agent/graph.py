from langgraph.graph import StateGraph, START, END

from backend.app.agent.state import AgentState
from backend.app.agent.retrieve_node import retrieve_node
from backend.app.agent.generate_node import generate_node
from backend.app.agent.evaluate_node import evaluate_node


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", END)

    return g.compile()