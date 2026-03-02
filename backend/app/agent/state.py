# backend/app/agent/state.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrievedArticle(BaseModel):
    """
    Structured representation of a retrieved constitutional article.
    Matches metadata structure used in build_index.py
    """

    page_content: str = Field(..., description="Full text of the retrieved article chunk")

    section_no_en: Optional[str] = Field(
        default=None,
        description="Article / Section number (e.g., 143)"
    )

    article_name_en: Optional[str] = Field(
        default=None,
        description="Title of the article"
    )

    part_name_en: Optional[str] = Field(
        default=None,
        description="Part name of the Constitution"
    )

    similarity_score: Optional[float] = Field(
        default=None,
        description="FAISS similarity score (if available)"
    )


class LatencyMetrics(BaseModel):
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    evaluation_time: Optional[float] = None
    total_time: Optional[float] = None


class AgentState(BaseModel):

    # ---------------------
    # Input
    # ---------------------
    user_query: str

    # ---------------------
    # Planning
    # ---------------------
    sub_queries: List[str] = Field(default_factory=list)

    # ---------------------
    # Retrieval
    # ---------------------
    retrieved_articles: List[RetrievedArticle] = Field(default_factory=list)

    # ---------------------
    # Generation
    # ---------------------
    draft_answer: Optional[str] = None

    # ---------------------
    # Evaluation
    # ---------------------
    citation_valid: Optional[bool] = None
    reliability_flag: Optional[bool] = None

    # ---------------------
    # Correction
    # ---------------------
    correction_triggered: bool = False
    corrected_answer: Optional[str] = None
    correction_attempts: int = 0

    # ---------------------
    # Final
    # ---------------------
    final_answer: Optional[str] = None

    # ---------------------
    # Profiling
    # ---------------------
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)

    # ---------------------
    # Debug
    # ---------------------
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True