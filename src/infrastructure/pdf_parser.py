import os
import pdfplumber
import pypdf
import pandas as pd
from typing import List, Dict


class PDFParser:
    """Extracts text and tables from PDF documents."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """Extracts text from the PDF file."""
        reader = pypdf.PdfReader(self.pdf_path)
        text = "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )
        return text

    def extract_tables(self) -> List[pd.DataFrame]:
        """Extracts tables from the PDF file and returns them as DataFrames."""
        tables = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    df = pd.DataFrame(table)
                    tables.append(df)
        return tables

    def parse(self) -> Dict[str, List[pd.DataFrame] | str]:
        """Extracts both text and tables from the PDF file."""
        return {"text": self.extract_text(), "tables": self.extract_tables()}


if __name__ == "__main__":
    pdf_path = "data/example.pdf"  # Replace with your actual PDF file
    parser = PDFParser(pdf_path)
    parsed_data = parser.parse()

    print("Extracted Text:")
    print(parsed_data["text"][:1000])  # Print first 1000 characters of text

    print("\nExtracted Tables:")
    for idx, table in enumerate(parsed_data["tables"]):
        print(f"\nTable {idx+1}:")
        print(table)
