from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator
import json

from backend.app.schemas.request import AskRequest
from backend.app.agent.state import AgentState
from backend.app.dependencies import get_graph

app = FastAPI(title="Constitutional Reliability RAG")

# -------------------------
# CORS
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health
# -------------------------

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# Structured Streaming
# -------------------------

def stream_response(state: AgentState) -> Generator[str, None, None]:

    # 1️⃣ Stream answer in small chunks (preserves formatting)
    answer_text = state.final_answer or ""

    chunk_size = 20  # small smooth streaming

    for i in range(0, len(answer_text), chunk_size):
        yield json.dumps({
            "type": "token",
            "data": answer_text[i:i+chunk_size]
        }) + "\n"

    # 2️⃣ Stream metadata including FULL article content
    yield json.dumps({
        "type": "metadata",
        "data": {
            "citation_valid": state.citation_valid,
            "reliability_flag": state.reliability_flag,
            "correction_triggered": state.correction_triggered,
            "correction_attempts": state.correction_attempts,
            "retrieved_articles": [
                {
                    "section_no_en": a.section_no_en,
                    "article_name_en": a.article_name_en,
                    "part_name_en": a.part_name_en,
                    "page_content": a.page_content,  # 🔥 FULL CONTENT FIX
                }
                for a in state.retrieved_articles
            ],
            "debug_info": state.debug_info,
        }
    }) + "\n"


@app.post("/ask/stream")
def ask_stream(request: AskRequest, graph=Depends(get_graph)):

    initial_state = AgentState(user_query=request.query)
    result = graph.invoke(initial_state)
    state = AgentState(**result)

    return StreamingResponse(
        stream_response(state),
        media_type="application/json"
    )