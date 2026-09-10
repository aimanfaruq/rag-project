# File: src/chunker.py
# Purpose:
# Split extracted PDF pages into smaller overlapping chunks
# so the RAG system can retrieve only relevant sections.

from pdf_loader import load_pdf_documents


def chunk_documents(documents, chunk_size=1200, overlap=200):
    """
    Split each PDF page into overlapping chunks.

    chunk_size:
        Maximum number of characters in one chunk.

    overlap:
        Number of characters shared with the next chunk
        to reduce context loss at chunk boundaries.
    """

    chunks = []

    for document in documents:
        text = document["text"]
        start = 0
        chunk_number = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": document["source"],
                        "page": document["page"],
                        "chunk": chunk_number,
                        "text": chunk_text,
                    }
                )

            # Stop when this chunk reaches the end of the page.
            if end >= len(text):
                break

            # Move forward while preserving overlap.
            start += chunk_size - overlap
            chunk_number += 1

    return chunks


if __name__ == "__main__":
    documents = load_pdf_documents()
    chunks = chunk_documents(documents)

    print(f"\nTotal pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        sample = chunks[0]

        print("\n--- SAMPLE CHUNK ---")
        print(f"Source: {sample['source']}")
        print(f"Page: {sample['page']}")
        print(f"Chunk: {sample['chunk']}")
        print(sample["text"][:1000])