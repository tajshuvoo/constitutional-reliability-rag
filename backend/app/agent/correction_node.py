from langchain_core.messages import SystemMessage, HumanMessage
from backend.app.agent.state import AgentState
from backend.app.agent.generate_node import llm, build_context


def correction_node(state: AgentState) -> AgentState:
    context = build_context(state.retrieved_articles)
    query = state.user_query
    previous_answer = state.draft_answer or ""
    failure_reason = state.debug_info.get("reason", "Reliability check failed.")

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a constitutional legal assistant performing self-correction."
                )
            ),
            HumanMessage(
                content=(
                    "Your previous answer failed reliability checks.\n\n"
                    f"Failure reason: {failure_reason}\n\n"
                    "PREVIOUS ANSWER:\n"
                    f"{previous_answer}\n\n"
                    "You must correct the answer.\n\n"
                    "STRICT REQUIREMENTS:\n"
                    "1. Every claim must include citation in format (Article NUMBER).\n"
                    "2. Only use the provided articles.\n"
                    "3. If not found, respond exactly with:\n"
                    "\"The answer is not found in the retrieved constitutional articles.\"\n\n"
                    f"ARTICLES:\n{context}\n\n"
                    f"QUESTION:\n{query}"
                )
            ),
        ]
    ).content.strip()

    state.corrected_answer = result
    state.draft_answer = result
    state.correction_triggered = True
    state.correction_attempts += 1
    return state