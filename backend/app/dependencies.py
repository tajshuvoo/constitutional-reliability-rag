from functools import lru_cache
from backend.app.agent.graph import build_graph


@lru_cache
def get_graph():
    return build_graph()