from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

from backend.app.agent.state import AgentState

load_dotenv()

_base_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    temperature=0,
    max_new_tokens=1000,
    top_p=0.9,
)

llm = ChatHuggingFace(llm=_base_llm)


def build_context(articles):
    blocks = []
    for a in articles:
        blocks.append(
            f"""
Part: {a.part_name_en}
Article Title: {a.article_name_en}
Section: {a.section_no_en}

{a.page_content}
""".strip()
        )
    return "\n\n---\n\n".join(blocks)


def generate_node(state: AgentState) -> AgentState:
    context = build_context(state.retrieved_articles)
    query = state.user_query

    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a constitutional text summarizer. "
                    "You must strictly restate constitutional provisions."
                )
            ),
            HumanMessage(
                content=(
                    "You are answering a constitutional legal question.\n\n"
                    "STRICT GENERATION RULES:\n"
                    "1. Only restate what is explicitly written in the provided articles.\n"
                    "2. Do NOT infer implications.\n"
                    "3. Do NOT analyze economic or political consequences.\n"
                    "4. Do NOT add interpretation.\n"
                    "5. Do NOT use evaluative language like 'enhances', 'strengthens', 'promotes', etc.\n"
                    "6. Every factual statement MUST include citation in format (Article NUMBER).\n"
                    "7. If the answer is not explicitly found, respond exactly with:\n"
                    "\"The answer is not found in the retrieved constitutional articles.\"\n\n"
                    f"CONSTITUTIONAL ARTICLES:\n{context}\n\n"
                    f"QUESTION:\n{query}\n\n"
                    "Provide a strictly textual, citation-grounded answer."
                )
            ),
        ]
    ).content.strip()

    state.draft_answer = result
    state.final_answer = result

    return state