# File: src/pdf_loader.py
# Purpose: Read all PDF manuals inside the documents folder
# and extract text page-by-page.

from pathlib import Path

from pypdf import PdfReader


# Project root: rag-project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local folder containing robot manuals
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def load_pdf_documents():
    """
    Load every PDF in the documents folder.

    Returns:
        A list of dictionaries containing:
        - source filename
        - page number
        - extracted text
    """
    pages = []

    # Find every PDF inside documents/
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            # Skip empty pages
            if not text:
                continue

            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


if __name__ == "__main__":
    documents = load_pdf_documents()

    print(f"\nTotal extracted pages: {len(documents)}")

    # Show a small sample so we can confirm PDF extraction works.
    if documents:
        first_page = documents[0]

        print("\n--- SAMPLE ---")
        print(f"Source: {first_page['source']}")
        print(f"Page: {first_page['page']}")
        print(first_page["text"][:1000])