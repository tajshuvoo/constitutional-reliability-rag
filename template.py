import os
from pathlib import Path


PROJECT_STRUCTURE = {
    "backend": {
        "app": {
            "__init__.py": "",
            "main.py": "",
            "config.py": "",
            "dependencies.py": "",
            "agent": {
                "__init__.py": "",
                "graph.py": "",
                "state.py": "",
                "planner_node.py": "",
                "retrieve_node.py": "",
                "generate_node.py": "",
                "evaluate_node.py": "",
                "correction_node.py": "",
                "finalize_node.py": "",
                "memory_node.py": "",
            },
            "retrieval": {
                "__init__.py": "",
                "pdf_loader.py": "",
                "chunker.py": "",
                "embedder.py": "",
                "vector_store.py": "",
                "retriever.py": "",
            },
            "evaluation": {
                "__init__.py": "",
                "recall.py": "",
                "faithfulness.py": "",
                "hallucination.py": "",
                "citation_check.py": "",
                "evaluator.py": "",
            },
            "profiling": {
                "__init__.py": "",
                "latency.py": "",
                "token_usage.py": "",
            },
            "services": {
                "__init__.py": "",
                "rag_pipeline.py": "",
                "logging_service.py": "",
            },
            "schemas": {
                "__init__.py": "",
                "request.py": "",
                "response.py": "",
                "metrics.py": "",
            },
        },
        "data": {
            "raw_pdf": {},
            "processed_chunks": {},
            "faiss_index": {},
            "logs": {},
        },
    },
    "frontend": {
        "__init__.py": "",
        "app.py": "",
        "components": {
            "__init__.py": "",
            "chat_ui.py": "",
            "metrics_panel.py": "",
            "retrieved_articles_view.py": "",
            "correction_view.py": "",
        },
        "utils.py": "",
    },
    "notebooks": {
        "index_building.ipynb": "",
        "evaluation_testing.ipynb": "",
    },
    "README.md": "# Constitutional Reliability RAG\n\nSelf-Correcting, Citation-Enforced Legal QA System.\n",
}


def create_structure(base_path: Path, structure: dict):
    for name, content in structure.items():
        path = base_path / name

        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


if __name__ == "__main__":
    create_structure(Path("."), PROJECT_STRUCTURE)
    print("✅ Project structure created successfully.")