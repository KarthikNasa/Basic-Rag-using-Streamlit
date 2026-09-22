# config.py

# ============================================================
# RAG CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Embedding Model
# ------------------------------------------------------------
# Converts text into vector embeddings.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Reranker Model
# ------------------------------------------------------------
# Used after ChromaDB retrieval to rank documents by relevance.
RERANKER_MODEL = "BAAI/bge-reranker-base"


# ------------------------------------------------------------
# LLM Model
# ------------------------------------------------------------
# Used to generate the final answer.
LLM_MODEL = "google/flan-t5-base"


# ------------------------------------------------------------
# ChromaDB Configuration
# ------------------------------------------------------------

# Folder where ChromaDB will store the vector database.
CHROMA_PATH = "./chroma_db"

# Name of the ChromaDB collection.
COLLECTION_NAME = "rag_documents"


# ------------------------------------------------------------
# Retrieval Configuration
# ------------------------------------------------------------

# Number of chunks initially retrieved from ChromaDB.
#
# Example:
# User question
#      ↓
# ChromaDB
#      ↓
# Top 10 chunks
#
TOP_K = 10


# ------------------------------------------------------------
# Reranking Configuration
# ------------------------------------------------------------

# Number of chunks kept after reranking.
#
# Example:
# Top 10 retrieved chunks
#          ↓
#       Reranker
#          ↓
#       Top 3 chunks
#
TOP_N = 3


# ------------------------------------------------------------
# Documents Configuration
# ------------------------------------------------------------

# Folder containing PDF/TXT/DOCX files.
DOCUMENTS_PATH = "./data/documents"
