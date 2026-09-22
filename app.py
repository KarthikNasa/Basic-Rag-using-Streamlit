# app.py

import streamlit as st

from rag import ask_rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f5f9ff;
        border-left: 5px solid #1f77b4;
        margin-bottom: 20px;
    }

    .source-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #fafafa;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📚 RAG Document Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your documents using '
    'Semantic Chunking, ChromaDB, Reranking '
    'and Hugging Face.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ RAG Configuration"
    )

    st.write(
        "Retrieval Pipeline"
    )

    st.markdown(
        """
        **1. Semantic Chunking**

        Documents are divided into
        meaningful semantic chunks.

        **2. Embeddings**

        Hugging Face converts chunks
        into vector representations.

        **3. ChromaDB**

        Retrieves the most relevant
        documents.

        **4. Top-K Retrieval**

        Retrieves the top 10 chunks.

        **5. Reranking**

        BGE CrossEncoder reranks
        the retrieved chunks.

        **6. Top-N**

        The best 3 chunks are selected.

        **7. LLM**

        Hugging Face generates
        the final answer.
        """
    )

    st.divider()

    st.info(
        "Top-K = 10\n\n"
        "Top-N = 3"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader(
    "Ask a Question"
)

question = st.text_area(
    "Enter your question below:",
    placeholder=(
        "Example:\n"
        "What is the company's leave policy?"
    ),
    height=120,
    label_visibility="collapsed"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "⚠️ Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # Show progress
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching documents, "
            "reranking results and generating answer..."
        ):

            try:

                result = ask_rag(
                    question
                )

                answer = result.get(
                    "answer",
                    ""
                )

                sources = result.get(
                    "sources",
                    []
                )

            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing your question."
                )

                st.exception(e)

                answer = None
                sources = []

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        if answer:

            st.divider()

            st.subheader(
                "💡 Answer"
            )

            st.markdown(
                f"""
                <div class="answer-box">

                {answer}

                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        st.subheader(
            "📚 Sources"
        )

        if not sources:

            st.info(
                "No source documents were returned."
            )

        else:

            st.caption(
                f"Top {len(sources)} "
                "reranked sources used for the answer."
            )

            for index, source in enumerate(
                sources,
                start=1
            ):

                filename = source.get(
                    "file",
                    "Unknown"
                )

                page = source.get(
                    "page",
                    "N/A"
                )

                score = source.get(
                    "score",
                    0
                )

                content = source.get(
                    "content",
                    ""
                )

                # ------------------------------------------------
                # Source expander
                # ------------------------------------------------

                with st.expander(
                    f"📄 Source {index} — "
                    f"{filename} "
                    f"(Page {page})"
                ):

                    col1, col2 = st.columns(
                        2
                    )

                    with col1:

                        st.write(
                            f"**File:** {filename}"
                        )

                    with col2:

                        st.write(
                            f"**Page:** {page}"
                        )

                    st.write(
                        f"**Reranker Score:** "
                        f"{score:.4f}"
                    )

                    st.divider()

                    st.write(
                        content
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Streamlit • "
    "Hugging Face • ChromaDB • "
    "Semantic Chunking • Reranking"
)
