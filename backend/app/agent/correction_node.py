from langchain_core.messages import SystemMessage, HumanMessage
from backend.app.agent.state import AgentState
from backend.app.agent.generate_node import llm, build_context


def correction_node(state: AgentState) -> AgentState:
    context = build_context(state.retrieved_articles)
    query = state.user_query
    previous_answer = state.draft_answer or ""

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a constitutional legal assistant performing strict correction."
                )
            ),
            HumanMessage(
                content=(
                    "Your previous answer failed reliability checks.\n\n"
                    "You must now produce a strictly grounded answer.\n\n"
                    "STRICT RULES:\n"
                    "1. Only state what is explicitly written in the constitutional articles.\n"
                    "2. Do NOT add implications, policy conclusions, or analysis.\n"
                    "3. Do NOT infer economic or political consequences.\n"
                    "4. Every factual statement must include citation like (Article NUMBER).\n"
                    "5. If the answer cannot be directly found, respond EXACTLY with:\n"
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