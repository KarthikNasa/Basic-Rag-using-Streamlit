# rag.py

from typing import List, Dict, Any

import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    LLM_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
    TOP_N,
)


# ============================================================
# RAG SYSTEM
# ============================================================

class RAGSystem:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING RAG SYSTEM")
        print("=" * 60)

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        # ----------------------------------------------------
        # Embedding Model
        # ----------------------------------------------------

        print()
        print(
            "Loading embedding model..."
        )

        print(
            EMBEDDING_MODEL
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print(
            "Embedding model loaded."
        )

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        print()
        print(
            "Connecting to ChromaDB..."
        )

        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
        )

        print(
            "ChromaDB connected."
        )

        # ----------------------------------------------------
        # Check ChromaDB
        # ----------------------------------------------------

        try:

            collection_data = (
                self.vectorstore.get()
            )

            document_count = len(
                collection_data.get(
                    "ids",
                    []
                )
            )

            print(
                f"Documents/chunks in ChromaDB: "
                f"{document_count}"
            )

        except Exception as e:

            print(
                f"Could not inspect ChromaDB: {e}"
            )

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        print()
        print(
            "Loading reranker..."
        )

        print(
            RERANKER_MODEL
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512
        )

        print(
            "Reranker loaded."
        )

        # ----------------------------------------------------
        # LLM Tokenizer
        # ----------------------------------------------------

        print()
        print(
            "Loading LLM tokenizer..."
        )

        print(
            LLM_MODEL
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                LLM_MODEL
            )
        )

        # ----------------------------------------------------
        # LLM Model
        # ----------------------------------------------------

        print(
            "Loading LLM model..."
        )

        self.llm = (
            AutoModelForSeq2SeqLM.from_pretrained(
                LLM_MODEL
            )
        )

        # Move model to CPU/GPU
        self.llm.to(
            self.device
        )

        print(
            "LLM loaded."
        )

        print()
        print("=" * 60)
        print("RAG SYSTEM READY")
        print("=" * 60)
        print()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str
    ):

        """
        Retrieve Top-K documents from ChromaDB.
        """

        print()
        print(
            "Retrieving documents..."
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Top-K: {TOP_K}"
        )

        documents = (
            self.vectorstore.similarity_search(
                question,
                k=TOP_K
            )
        )

        print(
            f"Retrieved {len(documents)} "
            f"documents."
        )

        return documents


    # ========================================================
    # RERANKING
    # ========================================================

    def rerank(
        self,
        question: str,
        documents
    ):

        """
        Rerank retrieved documents using
        a Hugging Face CrossEncoder.
        """

        print()
        print(
            "Reranking documents..."
        )

        if not documents:

            return []

        # ----------------------------------------------------
        # Create query-document pairs
        # ----------------------------------------------------

        pairs = []

        for document in documents:

            pairs.append(
                (
                    question,
                    document.page_content
                )
            )

        # ----------------------------------------------------
        # Calculate relevance scores
        # ----------------------------------------------------

        scores = (
            self.reranker.predict(
                pairs
            )
        )

        # ----------------------------------------------------
        # Combine documents and scores
        # ----------------------------------------------------

        ranked_documents = list(
            zip(
                documents,
                scores
            )
        )

        # ----------------------------------------------------
        # Sort highest score first
        # ----------------------------------------------------

        ranked_documents.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print(
            "Reranking completed."
        )

        # ----------------------------------------------------
        # Print ranking
        # ----------------------------------------------------

        for index, (
            document,
            score
        ) in enumerate(
            ranked_documents,
            start=1
        ):

            source = document.metadata.get(
                "source_file",
                "Unknown"
            )

            page = document.metadata.get(
                "page_number",
                "N/A"
            )

            print(
                f"{index}. "
                f"{source} | "
                f"Page: {page} | "
                f"Score: {float(score):.4f}"
            )

        return ranked_documents


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        documents
    ):

        """
        Build context from Top-N documents.
        """

        context_parts = []

        for index, (
            document,
            score
        ) in enumerate(
            documents,
            start=1
        ):

            source = document.metadata.get(
                "source_file",
                "Unknown"
            )

            page = document.metadata.get(
                "page_number",
                "N/A"
            )

            content = document.page_content

            context = f"""
SOURCE {index}
File: {source}
Page: {page}

Content:
{content}
"""

            context_parts.append(
                context
            )

        return "\n\n".join(
            context_parts
        )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question: str,
        documents
    ):

        """
        Generate final answer using
        the Hugging Face LLM.
        """

        print()
        print(
            "Generating answer..."
        )

        if not documents:

            return (
                "I could not find relevant "
                "information in the provided "
                "documents."
            )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = self.build_context(
            documents
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
information provided in the context.

Do not use outside knowledge.

Do not make up information.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided documents."

Keep the answer clear and concise.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=4,
            )

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        answer = answer.strip()

        print(
            "Answer generated."
        )

        return answer


    # ========================================================
    # PREPARE SOURCES
    # ========================================================

    def prepare_sources(
        self,
        documents
    ):

        """
        Prepare source information for Streamlit.
        """

        sources = []

        for document, score in documents:

            metadata = (
                document.metadata
            )

            source = {

                "file": metadata.get(
                    "source_file",
                    metadata.get(
                        "source",
                        "Unknown"
                    )
                ),

                "page": metadata.get(
                    "page_number",
                    "N/A"
                ),

                "score": float(
                    score
                ),

                "content": (
                    document.page_content
                )
            }

            sources.append(
                source
            )

        return sources


    # ========================================================
    # COMPLETE RAG PIPELINE
    # ========================================================

    def ask(
        self,
        question: str
    ) -> Dict[str, Any]:

        """
        Complete RAG pipeline:

        Question
            ↓
        ChromaDB
            ↓
        Top-K
            ↓
        Reranker
            ↓
        Top-N
            ↓
        LLM
            ↓
        Answer
        """

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:

            return {
                "answer": (
                    "Please enter a question."
                ),
                "sources": []
            }

        question = question.strip()

        if not question:

            return {
                "answer": (
                    "Please enter a question."
                ),
                "sources": []
            }

        # ----------------------------------------------------
        # STEP 1
        # Retrieve Top-K
        # ----------------------------------------------------

        documents = self.retrieve(
            question
        )

        if not documents:

            return {
                "answer": (
                    "I could not find any "
                    "relevant documents."
                ),
                "sources": []
            }

        # ----------------------------------------------------
        # STEP 2
        # Rerank
        # ----------------------------------------------------

        ranked_documents = self.rerank(
            question,
            documents
        )

        # ----------------------------------------------------
        # STEP 3
        # Select Top-N
        # ----------------------------------------------------

        top_documents = (
            ranked_documents[:TOP_N]
        )

        print()
        print(
            f"Selected Top-{TOP_N} "
            f"documents for LLM."
        )

        # ----------------------------------------------------
        # STEP 4
        # Generate Answer
        # ----------------------------------------------------

        answer = self.generate_answer(
            question,
            top_documents
        )

        # ----------------------------------------------------
        # STEP 5
        # Prepare Sources
        # ----------------------------------------------------

        sources = self.prepare_sources(
            top_documents
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }


# ============================================================
# CREATE RAG INSTANCE
# ============================================================

rag = None


def get_rag():

    """
    Creates the RAG system only once.
    """

    global rag

    if rag is None:

        rag = RAGSystem()

    return rag


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def ask_rag(
    question: str
):

    """
    Function used by Streamlit.
    """

    system = get_rag()

    return system.ask(
        question
    )
