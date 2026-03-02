import streamlit as st
import requests
import json

# BACKEND_URL = "http://127.0.0.1:8000/ask/stream"
BACKEND_URL = "https://constitutional-reliability-rag.onrender.com/ask/stream"

st.set_page_config(
    page_title="Constitutional Reliability RAG",
    layout="wide"
)

st.title("Constitutional Reliability RAG",text_alignment='center')

query = st.text_area(
    "Ask a constitutional question:",
    height=120
)

if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")
    else:

        answer_placeholder = st.empty()
        metadata_container = st.container()

        full_answer = ""

        try:
            response = requests.post(
                BACKEND_URL,
                json={"query": query},
                stream=True,
                timeout=300
            )

            for line in response.iter_lines():
                if line:

                    chunk = json.loads(line)

                    # --------------------
                    # STREAM ANSWER
                    # --------------------
                    if chunk["type"] == "token":

                        full_answer += chunk["data"]

                        answer_placeholder.markdown(
                            f"## Answer\n\n{full_answer}"
                        )

                    # --------------------
                    # METADATA
                    # --------------------
                    elif chunk["type"] == "metadata":

                        data = chunk["data"]

                        with metadata_container:

                            st.divider()
                            st.subheader("Reliability")

                            col1, col2, col3 = st.columns(3)

                            col1.metric("Citation Valid", data["citation_valid"])
                            col2.metric("Reliable", data["reliability_flag"])
                            col3.metric("Corrections", data["correction_attempts"])

                            st.divider()
                            st.subheader("Retrieved Articles")

                            for article in data["retrieved_articles"]:

                                with st.expander(
                                    f"Article {article['section_no_en']} — {article['article_name_en']}"
                                ):
                                    st.markdown(f"**Part:** {article['part_name_en']}")
                                    st.markdown("---")
                                    st.markdown(article["page_content"])  # 🔥 FULL CONTENT

                            if data["debug_info"]:
                                st.divider()
                                st.subheader("Debug Info")
                                st.json(data["debug_info"])

        except Exception as e:
            st.error("Failed to connect to backend.")
            st.exception(e)