import os
from typing import List, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker:
    """Handles text preprocessing and chunking using LangChain's text splitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
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

    def __init__(self, embedding_model: EmbeddingModel, index_path: str = None) -> None:
        """Initialize the FAISS vector store.

        Args:
            embedding_model: Model for generating embeddings
            index_path: Path to FAISS index. If None, uses default path
        """
        self.embedding_model = embedding_model
        self.vector_store = None
        # Make index path absolute and create directory if it doesn't exist
        if index_path:
            self.index_path = os.path.abspath(index_path)
        else:
            current_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            self.index_path = os.path.join(current_dir, "indexes", "faiss_index")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        logger.info(f"FAISS index path set to: {self.index_path}")

    def store_text_chunks(
        self, chunks: List[str], metadata_list: List[Dict[str, str]]
    ) -> None:
        """Stores text chunks in FAISS with metadata."""
        if not chunks or not metadata_list or len(chunks) != len(metadata_list):
            logger.error("Mismatch between chunks and metadata.")
            return

        # Load existing index if it exists
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}.")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embedding_model.embedding_model,
                allow_dangerous_deserialization=True,
            )
            # Add new chunks with metadata
            self.vector_store.add_texts(texts=chunks, metadatas=metadata_list)
        else:
            # Create a new index if it doesn't exist
            logger.info(f"Creating new FAISS index at {self.index_path}.")
            self.vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embedding_model.embedding_model,
                metadatas=metadata_list,
            )

        # Save the updated index
        self.vector_store.save_local(self.index_path)
        logger.info(f"FAISS index updated and saved at {self.index_path}.")

    def load_index(self) -> None:
        """Loads the FAISS index from disk."""
        try:
            logger.info(f"Attempting to load FAISS index from {self.index_path}")
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_model.embedding_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Successfully loaded FAISS index")
            else:
                logger.error(f"FAISS index not found at {self.index_path}")
                logger.info("Please ensure it is created and saved.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")

    def query_text(self, query: str, top_k: int) -> List[str]:
        """Retrieves the most relevant chunks from FAISS."""
        if self.vector_store is None:
            raise ValueError("FAISS index not loaded. Call 'load_index()' first.")

        logger.info(f"Querying FAISS index with query: {query}")
        results = self.vector_store.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} results.")

        # Return just the text content
        return [doc.page_content for doc in results]

    def list_documents(self) -> List[Dict[str, str]]:
        """Lists all documents stored in the FAISS index with their metadata."""
        if self.vector_store is None:
            self.load_index()
            if self.vector_store is None:
                logger.error("No index found.")
                return []

        # Get all document ids
        all_docs = self.vector_store.docstore._dict
        unique_docs = {}

        # Group by filename to avoid duplicates from chunks
        for doc_id, doc in all_docs.items():
            filename = doc.metadata.get("filename")
            if filename and filename not in unique_docs:
                unique_docs[filename] = {
                    "filename": filename,
                    "content_type": doc.metadata.get("content_type", "unknown"),
                }

        return list(unique_docs.values())

    def delete_document(self, filename: str) -> bool:
        """Deletes all chunks associated with a specific document filename.

        Args:
            filename (str): The filename to delete

        Returns:
            bool: True if document was found and deleted, False otherwise
        """
        if self.vector_store is None:
            self.load_index()
            if self.vector_store is None:
                logger.error("No index found.")
                return False

        # Find all document ids with matching filename
        docs_to_delete = []
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if doc.metadata.get("filename") == filename:
                docs_to_delete.append(doc_id)

        if not docs_to_delete:
            logger.info(f"No documents found with filename: {filename}")
            return False

        # Delete the documents
        for doc_id in docs_to_delete:
            del self.vector_store.docstore._dict[doc_id]
            # Also need to delete from the index
            self.vector_store.index_to_docstore_id.remove(doc_id)

        # Save the updated index
        self.vector_store.save_local(self.index_path)
        logger.info(f"Deleted document: {filename}")
        return True


class VectorStore:
    """Orchestrates text chunking, embedding, storage, and retrieval."""

    def __init__(
        self, api_key: str, chunk_size: int = 300, chunk_overlap: int = 50
    ) -> None:
        """Initialize the pipeline with OpenAI API key and FAISS index path.

        Args:
            api_key (str): OpenAI API key
            chunk_size (int, optional): Size of text chunks. Defaults to 300.
            chunk_overlap (int, optional): Overlap between chunks. Defaults to 50.
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = EmbeddingModel(api_key=api_key)
        self.vector_store = FaissVectorStore(self.embedding_model)

    def process_and_store_text(self, text: str, metadata: Dict[str, str]) -> None:
        """Processes text into chunks, generates embeddings, and stores them in FAISS with metadata."""
        chunks = self.chunker.chunk_text(text)
        metadata_list = [metadata for _ in chunks]  # Associate metadata with each chunk
        self.vector_store.store_text_chunks(chunks, metadata_list)
        logger.info(f"Stored {len(chunks)} text chunks in FAISS with metadata.")

    def retrieve_relevant_text(self, query: str, top_k: int) -> List[str]:
        """Retrieves relevant text chunks from FAISS along with metadata."""
        self.vector_store.load_index()
        results = self.vector_store.query_text(query, top_k)
        # Extract only the text content from the results
        return results

    def list_stored_documents(self) -> List[Dict[str, str]]:
        """Lists all documents stored in the vector store."""
        return self.vector_store.list_documents()

    def delete_document(self, filename: str) -> bool:
        """Deletes a document and all its chunks from the vector store."""
        return self.vector_store.delete_document(filename)
