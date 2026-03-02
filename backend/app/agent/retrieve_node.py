# backend/app/agent/retrieve_node.py

from typing import List

from backend.app.agent.state import AgentState, RetrievedArticle
from backend.app.retrieval.retriever import ConstitutionRetriever


retriever = ConstitutionRetriever()


def retrieve_node(state: AgentState) -> AgentState:

    queries = state.sub_queries or [state.user_query]

    collected: List[RetrievedArticle] = []

    for query in queries:

        docs = retriever.retrieve(query)

        for doc in docs:
            collected.append(
                RetrievedArticle(
                    page_content=doc.page_content,
                    section_no_en=doc.metadata.get("section_no_en"),
                    article_name_en=doc.metadata.get("article_name_en"),
                    part_name_en=doc.metadata.get("part_name_en"),
                    similarity_score=None,
                )
            )

    # Deduplicate by section number
    unique = {}
    for article in collected:
        unique[article.section_no_en] = article

    state.retrieved_articles = list(unique.values())

    return state