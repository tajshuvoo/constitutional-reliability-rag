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
            SystemMessage(content="You are a Bangladesh constitutional legal assistant."),
            HumanMessage(
                content=(
                    "You are answering a constitutional legal question.\n\n"
                    "STRICT RULES:\n"
                    "1. You MUST answer only from the provided articles.\n"
                    "2. Every factual claim MUST include a citation in this exact format: (Article NUMBER).\n"
                    "3. Citation must be inside parentheses.\n"
                    "4. If the answer is not found, respond exactly with:\n"
                    "\"The answer is not found in the retrieved constitutional articles.\"\n\n"
                    "Do NOT provide any information without citation.\n\n"
                    f"CONSTITUTIONAL ARTICLES:\n{context}\n\n"
                    f"QUESTION:\n{query}\n\n"
                    "Provide a legally grounded answer with proper citations."
                )
            ),
        ]
    ).content.strip()

    state.draft_answer = result
    state.final_answer = result

    return state