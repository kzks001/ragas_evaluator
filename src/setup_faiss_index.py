import os
from loguru import logger

from infrastructure.pdf_parser import PDFParser
from infrastructure.text_embedding_pipeline import VectorStore

api_key = os.getenv("OPENAI_API_KEY")
directory_path = "data/pdf_docs"  # Ensure this path points to the folder containing PDFs


def _store_in_faiss(faiss_vector_store, content: str, filename: str, content_type: str):
    """Stores extracted text or table data into the FAISS index with metadata."""
    if content:
        metadata = {"filename": filename, "content_type": content_type}
        logger.info(f"Storing {len(content)} characters from {filename} ({content_type}) into FAISS...")
        faiss_vector_store.process_and_store_text(content, metadata)


def setup_faiss_index():
    """Parses multiple PDF documents and loads their text, tables, and metadata into the FAISS index."""
    # Initialize FAISS vector store
    faiss_vector_store = VectorStore(api_key)

    if not os.path.isdir(directory_path):
        logger.error(f"Directory {directory_path} does not exist. Exiting.")
        return

    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]

    if not pdf_files:
        logger.info("No PDF files found in the directory. Exiting.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        logger.info(f"Parsing PDF: {pdf_path}")

        # Parse PDF
        pdf_doc = PDFParser(pdf_path)
        parsed_data = pdf_doc.parse()
        text = parsed_data.get("text", "")
        tables = parsed_data.get("tables", [])

        if not text and not tables:
            logger.info(f"No content found in {pdf_file}. Skipping.")
            continue

        # Store text and tables
        _store_in_faiss(faiss_vector_store, text, pdf_file, "text")
        for idx, table in enumerate(tables):
            table_text = str(table)  # Convert table data to a string
            _store_in_faiss(faiss_vector_store, table_text, pdf_file, f"table_{idx + 1}")

    logger.info("FAISS index setup complete for all PDFs.")


if __name__ == "__main__":
    setup_faiss_index()
