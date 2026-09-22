# ingest.py

import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from config import (
    DOCUMENTS_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================

def load_documents():

    print("=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)

    documents = []

    # Check whether document folder exists
    if not os.path.exists(DOCUMENTS_PATH):

        print(
            f"ERROR: Document folder does not exist:"
        )

        print(
            DOCUMENTS_PATH
        )

        return documents

    # Get all files
    files = os.listdir(DOCUMENTS_PATH)

    if not files:

        print(
            "No files found in the document folder."
        )

        return documents

    # Process each file
    for filename in files:

        file_path = os.path.join(
            DOCUMENTS_PATH,
            filename
        )

        # Skip directories
        if not os.path.isfile(file_path):

            continue

        try:

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            if filename.lower().endswith(".pdf"):

                print(
                    f"Loading PDF: {filename}"
                )

                loader = PyPDFLoader(
                    file_path
                )

                docs = loader.load()

                documents.extend(docs)


            # ------------------------------------------------
            # TXT
            # ------------------------------------------------

            elif filename.lower().endswith(".txt"):

                print(
                    f"Loading TXT: {filename}"
                )

                loader = TextLoader(
                    file_path,
                    encoding="utf-8"
                )

                docs = loader.load()

                documents.extend(docs)


            # ------------------------------------------------
            # DOCX
            # ------------------------------------------------

            elif filename.lower().endswith(".docx"):

                print(
                    f"Loading DOCX: {filename}"
                )

                loader = Docx2txtLoader(
                    file_path
                )

                docs = loader.load()

                documents.extend(docs)


            # ------------------------------------------------
            # Unsupported file
            # ------------------------------------------------

            else:

                print(
                    f"Skipping unsupported file: "
                    f"{filename}"
                )

        except Exception as e:

            print(
                f"ERROR loading {filename}: {e}"
            )

    print()
    print(
        f"Total documents/pages loaded: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# 2. CREATE SEMANTIC CHUNKS
# ============================================================

def create_semantic_chunks(documents):

    print()
    print("=" * 60)
    print("CREATING SEMANTIC CHUNKS")
    print("=" * 60)

    if not documents:

        print(
            "No documents available for chunking."
        )

        return []

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "Loading embedding model:"
    )

    print(
        EMBEDDING_MODEL
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Semantic Chunker
    # --------------------------------------------------------

    print(
        "Initializing SemanticChunker..."
    )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    print(
        "Splitting documents semantically..."
    )

    chunks = splitter.split_documents(
        documents
    )

    print()
    print(
        f"Total semantic chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# 3. ADD EXTRA METADATA
# ============================================================

def add_metadata(chunks):

    print()
    print("=" * 60)
    print("ADDING METADATA")
    print("=" * 60)

    for index, chunk in enumerate(chunks):

        # Add chunk ID
        chunk.metadata["chunk_id"] = index

        # Make source filename easier to display
        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        chunk.metadata["source_file"] = os.path.basename(
            source
        )

        # Make page number human-friendly
        if "page" in chunk.metadata:

            chunk.metadata["page_number"] = (
                chunk.metadata["page"] + 1
            )

        else:

            chunk.metadata["page_number"] = "N/A"

    print(
        "Metadata added successfully."
    )

    return chunks


# ============================================================
# 4. CREATE CHROMADB
# ============================================================

def create_vector_database(chunks):

    print()
    print("=" * 60)
    print("CREATING CHROMADB")
    print("=" * 60)

    if not chunks:

        print(
            "No chunks available."
        )

        return None

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "Loading embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    print(
        f"ChromaDB path: {CHROMA_PATH}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # Create / connect to ChromaDB
    # --------------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # --------------------------------------------------------
    # Remove existing collection documents
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We DO NOT delete the chroma_db folder.
    #
    # This avoids Windows:
    #
    # PermissionError: [WinError 5]
    #
    # Instead, we clear the collection through ChromaDB.
    # --------------------------------------------------------

    try:

        existing_data = vectorstore.get()

        existing_ids = existing_data.get(
            "ids",
            []
        )

        if existing_ids:

            print(
                f"Removing {len(existing_ids)} "
                f"existing chunks..."
            )

            vectorstore.delete(
                ids=existing_ids
            )

        else:

            print(
                "No existing chunks found."
            )

    except Exception as e:

        print(
            "Warning while clearing existing "
            f"collection: {e}"
        )

    # --------------------------------------------------------
    # Add new documents
    # --------------------------------------------------------

    print()
    print(
        "Creating embeddings and storing "
        "in ChromaDB..."
    )

    vectorstore.add_documents(
        documents=chunks
    )

    print()
    print(
        "ChromaDB created successfully."
    )

    print(
        f"Total chunks stored: {len(chunks)}"
    )

    return vectorstore


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Step 1: Load documents
    # --------------------------------------------------------

    documents = load_documents()

    if not documents:

        print()
        print(
            "No documents were loaded."
        )

        print(
            f"Please place PDF/TXT/DOCX files inside:"
        )

        print(
            DOCUMENTS_PATH
        )

        return

    # --------------------------------------------------------
    # Step 2: Semantic chunking
    # --------------------------------------------------------

    chunks = create_semantic_chunks(
        documents
    )

    if not chunks:

        print(
            "No chunks were created."
        )

        return

    # --------------------------------------------------------
    # Step 3: Add metadata
    # --------------------------------------------------------

    chunks = add_metadata(
        chunks
    )

    # --------------------------------------------------------
    # Step 4: Create ChromaDB
    # --------------------------------------------------------

    create_vector_database(
        chunks
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print()

    print(
        "You can now run:"
    )

    print(
        "streamlit run app.py"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
