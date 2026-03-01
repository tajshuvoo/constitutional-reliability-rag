# backend/app/services/logging_service.py

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from backend.app.agent.state import AgentState


LOG_PATH = Path("backend/data/logs")
LOG_FILE = LOG_PATH / "query_logs.jsonl"


def _ensure_log_dir():
    LOG_PATH.mkdir(parents=True, exist_ok=True)


def _extract_retrieved_articles(state: AgentState):
    return [
        {
            "section_no_en": article.section_no_en,
            "article_name_en": article.article_name_en,
            "part_name_en": article.part_name_en,
        }
        for article in state.retrieved_articles
    ]


def log_query(state: AgentState) -> None:
    """
    Append one structured log entry to JSONL file.
    """

    _ensure_log_dir()

    log_entry: Dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_query": state.user_query,
        "retrieved_articles": _extract_retrieved_articles(state),
        "draft_answer": state.draft_answer,
        "corrected_answer": state.corrected_answer,
        "final_answer": state.final_answer,
        "citation_valid": state.citation_valid,
        "reliability_flag": state.reliability_flag,
        "correction_triggered": state.correction_triggered,
        "correction_attempts": state.correction_attempts,
        "debug_info": state.debug_info,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")