from backend.app.agent.graph import build_graph
from backend.app.agent.state import AgentState
from backend.app.services.logging_service import log_query

app = build_graph()


def ask(query: str) -> AgentState:
    initial_state = AgentState(user_query=query)

    result = app.invoke(initial_state)
    final_state = AgentState(**result)

    log_query(final_state)

    return final_state


if __name__ == "__main__":
    state = ask("Critically analyze the impact of Article 143 on economic sovereignty.")

    print("\n===== FINAL OUTPUT =====\n")
    print("Answer:\n", state.final_answer)
    print("\nCitation Valid:", state.citation_valid)
    print("Reliability Flag:", state.reliability_flag)
    print("Correction Triggered:", state.correction_triggered)
    print("Correction Attempts:", state.correction_attempts)
    print("Debug Info:", state.debug_info)