from backend.app.retrieval.retriever import ConstitutionRetriever

if __name__ == "__main__":

    retriever = ConstitutionRetriever()

    query = "territory of Bangladesh"

    results = retriever.retrieve(query)

    for i, doc in enumerate(results, 1):
        print("\n===================")
        print(f"Result {i}")
        print("Section:", doc.metadata.get("section_no_en"))
        print("Title:", doc.metadata.get("article_name_en"))
        print(doc.page_content)