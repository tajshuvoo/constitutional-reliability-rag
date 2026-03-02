# backend/app/retrieval/build_index.py

from pathlib import Path
import json
import os

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS


JSON_PATH = Path("backend/data/bangladesh-constitution.json")
INDEX_PATH = "backend/data/processed_chunks/faiss_index"


def build_faiss_index():
    print("Loading JSON...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for entry in data:
        content = entry.get("content", "").strip()
        if not content:
            continue

        structured_text = f"""
Part: {entry.get("part_name_en")}
Article Title: {entry.get("article_name_en")}
Section: {entry.get("section_no_en")}

{content}
""".strip()

        metadata = {
            "section_no_en": entry.get("section_no_en"),
            "article_name_en": entry.get("article_name_en"),
            "part_name_en": entry.get("part_name_en"),
        }

        documents.append(
            Document(
                page_content=structured_text,
                metadata=metadata
            )
        )

    print(f"Total documents prepared: {len(documents)}")

    if not documents:
        raise ValueError("No documents found. Check JSON path.")

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Ensure directory exists
    os.makedirs("backend/data/processed_chunks", exist_ok=True)

    vectorstore.save_local(INDEX_PATH)

    print(f"FAISS index saved at: {INDEX_PATH}")


if __name__ == "__main__":
    build_faiss_index()