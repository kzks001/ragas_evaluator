import os

from loguru import logger 

from infrastructure.pdf_parser import PDFParser
from infrastructure.text_embedding_pipeline import VectorStore

api_key = os.getenv("OPENAI_API_KEY")
file_path = "../data/example.pdf"


def setup_faiss_index():
    """Parses the PDF document and loads its text and tables into the FAISS index."""
    # Initialize FAISS vector store
    faiss_vector_store = VectorStore(api_key)

    logger.info(f"Parsing PDF: {file_path}")

    # Parse PDF
    pdf_doc = PDFParser(file_path)
    parsed_data = pdf_doc.parse()
    text = parsed_data.get("text", "")
    tables = parsed_data.get("tables", [])

    if not text and not tables:
        logger.info("No content found in the PDF. Exiting.")
        return

    # Process and store text
    if text:
        logger.info(f"Storing {len(text)} characters of extracted text into FAISS...")
        faiss_vector_store.process_and_store_text(text)

    # Process and store tables (converting to text)
    for idx, table in enumerate(tables):
        table_text = str(table)  # Convert table data to a string
        logger.info(f"Storing table {idx + 1} in FAISS...")
        faiss_vector_store.process_and_store_text(table_text)

    logger.info("FAISS index setup complete.")


if __name__ == "__main__":
    setup_faiss_index()
