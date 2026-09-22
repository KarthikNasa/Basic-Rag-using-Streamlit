# ingest.py

import os
import shutil

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_experimental.text_splitter import SemanticChunker

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from config import (
    DOCUMENTS_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):

        file_path = os.path.join(
            DOCUMENTS_PATH,
            filename
        )

        if filename.lower().endswith(".pdf"):

            print(f"Loading PDF: {filename}")

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            documents.extend(docs)

        elif filename.lower().endswith(".txt"):

            print(f"Loading TXT: {filename}")

            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

            docs = loader.load()

            documents.extend(docs)

        elif filename.lower().endswith(".docx"):

            print(f"Loading DOCX: {filename}")

            loader = Docx2txtLoader(file_path)

            docs = loader.load()

            documents.extend(docs)

    return documents


def create_semantic_chunks(documents):

    print("Creating semantic chunks...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def create_vector_database(chunks):

    print("Creating embeddings and storing in ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Delete existing database
    if os.path.exists(CHROMA_PATH):

        print("Removing old ChromaDB...")

        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("ChromaDB created successfully.")

    return vectorstore


def main():

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    documents = load_documents()

    if not documents:

        print("No documents found.")

        print(
            f"Please put PDF/TXT/DOCX files inside: "
            f"{DOCUMENTS_PATH}"
        )

        return

    print(
        f"Loaded {len(documents)} document pages/files."
    )

    chunks = create_semantic_chunks(documents)

    create_vector_database(chunks)

    print("=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

