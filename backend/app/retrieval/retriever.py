# backend/app/retrieval/retriever.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings
INDEX_PATH = "backend/data/processed_chunks/faiss_index"


class ConstitutionRetriever:
    def __init__(self):

        embeddings = HuggingFaceEndpointEmbeddings(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
        )

        # Load persistent FAISS
        self.vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Semantic retriever
        semantic = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # BM25 retriever (keyword)
        docs = list(self.vectorstore.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = 5

        # Hybrid
        self.retriever = EnsembleRetriever(
            retrievers=[semantic, bm25],
            weights=[0.7, 0.3],
        )

    def retrieve(self, query: str):
        return self.retriever.invoke(query)