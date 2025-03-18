import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker:
    """Handles text preprocessing and chunking using LangChain's text splitter."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        """Initialize the text chunker with chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into smaller chunks."""
        return self.text_splitter.split_text(text)


class EmbeddingModel:
    """Handles text embeddings using OpenAI's Embeddings API."""

    def __init__(self, api_key: str) -> None:
        """Initialize the embedding model with the given API key."""
        self.embedding_model = OpenAIEmbeddings(api_key=api_key)

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        return self.embedding_model.embed_query(text)


class FaissVectorStore:
    """Handles FAISS-based vector storage and retrieval."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index_path: str = "../ragas_evaluation/faiss_index",
    ) -> None:
        """Initialize FAISS store with an embedding model and index path."""
        self.embedding_model = embedding_model
        self.index_path = os.path.abspath(index_path)
        self.vector_store = None

    def store_text_chunks(self, chunks: List[str]) -> None:
        """Stores the text chunks in FAISS after generating embeddings."""
        # Load existing index if it exists
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}.")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embedding_model.embedding_model,
                allow_dangerous_deserialization=True,
            )
            # Add new chunks to the existing index
            self.vector_store.add_texts(chunks)
        else:
            # Create a new index if it doesn't exist
            logger.info(f"Creating new FAISS index at {self.index_path}.")
            self.vector_store = FAISS.from_texts(
                chunks, self.embedding_model.embedding_model
            )

        # Save the updated index
        self.vector_store.save_local(self.index_path)
        logger.info(f"FAISS index updated and saved at {self.index_path}.")

    def load_index(self) -> None:
        """Loads the FAISS index from disk."""
        if os.path.exists(self.index_path):
            logger.info(f"Loading FAISS index from {self.index_path}.")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embedding_model.embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("FAISS index not found. Please ensure it is created and saved.")

    def query_text(self, query: str, top_k: int) -> List[str]:
        """Retrieves the most relevant chunks from FAISS."""
        if self.vector_store is None:
            raise ValueError("FAISS index not loaded. Call 'load_index()' first.")

        logger.info(f"Querying FAISS index with query: {query}")
        results = self.vector_store.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} results.")
        return [doc.page_content for doc in results]


class VectorStore:
    """Orchestrates text chunking, embedding, storage, and retrieval."""

    def __init__(self, api_key: str) -> None:
        """Initialize the pipeline with OpenAI API key and FAISS index path."""
        self.chunker = TextChunker()
        self.embedding_model = EmbeddingModel(api_key=api_key)
        self.vector_store = FaissVectorStore(self.embedding_model)

    def process_and_store_text(self, text: str) -> None:
        """Processes text into chunks, generates embeddings, and stores them in FAISS."""
        chunks = self.chunker.chunk_text(text)
        self.vector_store.store_text_chunks(chunks)
        logger.info(f"Stored {len(chunks)} text chunks in FAISS.")

    def retrieve_relevant_text(self, query: str, top_k: int) -> List[str]:
        """Retrieves relevant text chunks from FAISS based on a query."""
        self.vector_store.load_index()
        results = self.vector_store.query_text(query, top_k)
        return results
