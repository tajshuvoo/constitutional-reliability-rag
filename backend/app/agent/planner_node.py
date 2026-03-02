# backend/app/agent/planner_node.py

from typing import List
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from backend.app.agent.state import AgentState
from backend.app.agent.generate_node import llm


# -----------------------------
# Schema
# -----------------------------
class PlannerDecision(BaseModel):
    sub_queries: List[str] = Field(
        ...,
        description="Minimal atomic constitutional sub-questions."
    )


planner_parser = PydanticOutputParser(
    pydantic_object=PlannerDecision
)


PLANNER_PROMPT = PromptTemplate(
    template=(
        "You are a constitutional question decomposition planner.\n\n"
        "Break the QUESTION into atomic constitutional sub-queries.\n\n"
        "Rules:\n"
        "1. If it mentions multiple Articles, split them.\n"
        "2. If it compares Articles, create separate sub-queries.\n"
        "3. If already atomic, return it unchanged.\n"
        "4. Do NOT answer the question.\n"
        "5. Return strictly valid JSON.\n\n"
        "QUESTION:\n{question}\n\n"
        "{format_instructions}"
    ),
    input_variables=["question"],
    partial_variables={
        "format_instructions": planner_parser.get_format_instructions()
    }
)


def planner_node(state: AgentState) -> AgentState:

    question = state.user_query

    prompt = PLANNER_PROMPT.invoke({"question": question})

    for _ in range(3):

        result = llm.invoke([
            SystemMessage(content="Return strictly valid JSON."),
            HumanMessage(content=prompt.to_string())
        ]).content

        try:
            decision = planner_parser.parse(result)
            state.sub_queries = decision.sub_queries
            return state

        except Exception:
            continue

    # Fallback
    state.sub_queries = [question]
    return state