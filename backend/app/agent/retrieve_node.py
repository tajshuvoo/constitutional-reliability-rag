from typing import List

from backend.app.agent.state import AgentState, RetrievedArticle
from backend.app.retrieval.retriever import ConstitutionRetriever


# Initialize once (important)
retriever = ConstitutionRetriever()


def retrieve_node(state: AgentState) -> AgentState:
    query = state.user_query

    docs = retriever.retrieve(query)

    articles: List[RetrievedArticle] = []

    for doc in docs:
        articles.append(
            RetrievedArticle(
                page_content=doc.page_content,
                section_no_en=doc.metadata.get("section_no_en"),
                article_name_en=doc.metadata.get("article_name_en"),
                part_name_en=doc.metadata.get("part_name_en"),
                similarity_score=None,  # EnsembleRetriever doesn't provide score
            )
        )

    state.retrieved_articles = articles

    return state