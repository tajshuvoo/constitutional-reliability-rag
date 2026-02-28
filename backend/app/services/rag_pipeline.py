from backend.app.agent.graph import build_graph
from backend.app.agent.state import AgentState

app = build_graph()

if __name__ == "__main__":
    state = AgentState(user_query="What does Article 143 state?")
    result = app.invoke(state)

    final_state = AgentState(**result)

    print("Answer:\n", final_state.final_answer)
    print("\nCitation Valid:", final_state.citation_valid)
    print("Reliability Flag:", final_state.reliability_flag)
    print("Debug Info:", final_state.debug_info)