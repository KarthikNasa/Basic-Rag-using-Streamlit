# rag.py

from typing import List, Dict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

import torch

from config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    LLM_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
    TOP_N,
)


class RAGSystem:

    def __init__(self):

        print("Loading embedding model...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Loading ChromaDB...")

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

        print("Loading reranker...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("Loading LLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL
        )

        self.llm = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.llm.to(self.device)

        print("RAG system loaded.")

    def retrieve(self, question: str):

        documents = self.vectorstore.similarity_search(
            question,
            k=TOP_K
        )

        return documents

    def rerank(
        self,
        question: str,
        documents
    ):

        if not documents:
            return []

        pairs = [
            (
                question,
                document.page_content
            )
            for document in documents
        ]

        scores = self.reranker.predict(
            pairs
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    def generate_answer(
        self,
        question: str,
        documents
    ):

        context = "\n\n".join(
            [
                document.page_content
                for document, score in documents
            ]
        )

        prompt = f"""
You are a helpful question answering assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer

    def ask(self, question: str):

        # --------------------------------
        # STEP 1: Retrieve Top-K
        # --------------------------------

        documents = self.retrieve(
            question
        )

        # --------------------------------
        # STEP 2: Rerank
        # --------------------------------

        ranked_documents = self.rerank(
            question,
            documents
        )

        # --------------------------------
        # STEP 3: Select Top-N
        # --------------------------------

        top_documents = ranked_documents[
            :TOP_N
        ]

        # --------------------------------
        # STEP 4: Generate Answer
        # --------------------------------

        answer = self.generate_answer(
            question,
            top_documents
        )

        # --------------------------------
        # STEP 5: Prepare Sources
        # --------------------------------

        sources = []

        for document, score in top_documents:

            metadata = document.metadata

            source = {
                "file": metadata.get(
                    "source",
                    "Unknown"
                ),

                "page": metadata.get(
                    "page",
                    "N/A"
                ),

                "score": float(score),

                "content": document.page_content
            }

            sources.append(source)

        return {
            "answer": answer,
            "sources": sources
        }


# Create one RAG instance
rag = RAGSystem()


def ask_rag(question: str):

    return rag.ask(question)

