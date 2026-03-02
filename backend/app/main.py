from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator

from backend.app.schemas.request import AskRequest
from backend.app.schemas.response import AskResponse, RetrievedArticleResponse
from backend.app.agent.state import AgentState
from backend.app.dependencies import get_graph

app = FastAPI(title="Constitutional Reliability RAG")

# -------------------------
# CORS Configuration
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (safe for public API)
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
# Ask Endpoint
# -------------------------

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(
    request: AskRequest,
    graph=Depends(get_graph)
):

    initial_state = AgentState(user_query=request.query)
    result = graph.invoke(initial_state)
    state = AgentState(**result)

    return AskResponse(
        answer=state.final_answer,
        citation_valid=state.citation_valid,
        reliability_flag=state.reliability_flag,
        correction_triggered=state.correction_triggered,
        correction_attempts=state.correction_attempts,
        retrieved_articles=[
            RetrievedArticleResponse(
                section_no_en=a.section_no_en,
                article_name_en=a.article_name_en,
                part_name_en=a.part_name_en,
            )
            for a in state.retrieved_articles
        ],
        debug_info=state.debug_info,
    )

# -------------------------
# Streaming
# -------------------------

def stream_answer(text: str) -> Generator[str, None, None]:
    for token in text.split():
        yield token + " "

@app.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    graph=Depends(get_graph)
):

    initial_state = AgentState(user_query=request.query)
    result = graph.invoke(initial_state)
    state = AgentState(**result)

    return StreamingResponse(
        stream_answer(state.final_answer),
        media_type="text/plain"
    )