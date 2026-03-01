# backend/app/evaluation/faithfulness.py

from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from backend.app.agent.state import AgentState
from backend.app.agent.generate_node import llm


# -----------------------------
# Schema
# -----------------------------
class FaithfulnessDecision(BaseModel):
    verdict: Literal["SUPPORTED", "UNSUPPORTED"] = Field(
        ...,
        description="Whether the answer is fully supported by the provided context."
    )


faithfulness_parser = PydanticOutputParser(
    pydantic_object=FaithfulnessDecision
)


# -----------------------------
# Prompt
# -----------------------------
FAITHFULNESS_PROMPT = PromptTemplate(
    template=(
        "You are a strict constitutional fact-checker.\n\n"
        "Determine whether the ANSWER is fully supported by the CONTEXT.\n\n"
        "If every claim in the answer is directly supported by the context, return SUPPORTED.\n"
        "If the answer contains any unsupported claim, interpretation, or addition, return UNSUPPORTED.\n\n"
        "Return strictly valid JSON.\n"
        "Do not include markdown.\n"
        "Do not include explanations.\n\n"
        "CONTEXT:\n{context}\n\n"
        "ANSWER:\n{answer}\n\n"
        "{format_instructions}"
    ),
    input_variables=["context", "answer"],
    partial_variables={
        "format_instructions": faithfulness_parser.get_format_instructions()
    }
)


# -----------------------------
# Faithfulness Function
# -----------------------------
def check_faithfulness(state: AgentState) -> dict:
    context = "\n\n".join(
        article.page_content for article in state.retrieved_articles
    )

    answer = state.draft_answer or ""

    prompt = FAITHFULNESS_PROMPT.invoke({
        "context": context,
        "answer": answer
    })

    for attempt in range(3):

        result = llm.invoke([
            SystemMessage(content="Return strictly valid JSON."),
            HumanMessage(content=prompt.to_string())
        ]).content

        try:
            decision = faithfulness_parser.parse(result)

            return {
                "faithful": decision.verdict == "SUPPORTED",
                "faithfulness_reason": f"LLM verdict: {decision.verdict}"
            }

        except Exception as e:
            print(f"[Faithfulness Parse Failed - Attempt {attempt+1}] {e}")

    # Fail-safe: mark as unsafe if parsing fails
    return {
        "faithful": False,
        "faithfulness_reason": "Parsing failed. Marked as UNSUPPORTED."
    }