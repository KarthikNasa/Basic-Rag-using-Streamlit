# app.py

import streamlit as st

from rag import ask_rag


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# -----------------------------------------
# Title
# -----------------------------------------

st.title("📚 RAG Document Assistant")

st.write(
    "Ask questions about your documents using "
    "semantic search, reranking and Hugging Face."
)


# -----------------------------------------
# Question input
# -----------------------------------------

question = st.text_area(
    "Ask your question",
    placeholder=(
        "Example: What is the company's "
        "leave policy?"
    ),
    height=100
)


# -----------------------------------------
# Ask button
# -----------------------------------------

if st.button(
    "🔍 Ask Question",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = ask_rag(
                    question
                )

                # -----------------------------
                # Answer
                # -----------------------------

                st.subheader("💡 Answer")

                st.write(
                    result["answer"]
                )

                # -----------------------------
                # Sources
                # -----------------------------

                st.subheader(
                    "📚 Sources"
                )

                if not result["sources"]:

                    st.write(
                        "No sources found."
                    )

                else:

                    for index, source in enumerate(
                        result["sources"],
                        start=1
                    ):

                        with st.expander(
                            f"Source {index}: "
                            f"{source['file']}"
                        ):

                            st.write(
                                f"**Page:** "
                                f"{source['page']}"
                            )

                            st.write(
                                f"**Reranker score:** "
                                f"{source['score']:.4f}"
                            )

                            st.write(
                                source["content"]
                            )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

