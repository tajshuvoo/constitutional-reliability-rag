from typing import List, Optional
from pydantic import BaseModel


class RetrievedArticleResponse(BaseModel):
    section_no_en: Optional[str]
    article_name_en: Optional[str]
    part_name_en: Optional[str]


class AskResponse(BaseModel):
    answer: str
    citation_valid: bool
    reliability_flag: bool
    correction_triggered: bool
    correction_attempts: int
    retrieved_articles: List[RetrievedArticleResponse]
    debug_info: dict