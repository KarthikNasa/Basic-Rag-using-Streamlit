# config.py

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Local Hugging Face LLM
LLM_MODEL = "google/flan-t5-base"

# ChromaDB location
CHROMA_PATH = "./chroma_db"

# Collection name
COLLECTION_NAME = "rag_documents"

# Number of documents retrieved from ChromaDB
TOP_K = 10

# Number of documents sent to the LLM after reranking
TOP_N = 3

# Documents directory
DOCUMENTS_PATH = "./data/documents"

