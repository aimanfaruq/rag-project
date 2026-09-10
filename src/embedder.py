# File: src/embedder.py
# Purpose:
# Convert document chunks into vector embeddings and save them locally.
#
# We save the embeddings so we do NOT have to call the API
# every time we run the RAG application.

import time

from openai import OpenAI, RateLimitError

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from pdf_loader import load_pdf_documents
from chunker import chunk_documents


# Load OPENAI_API_KEY from .env
load_dotenv()

client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Create data/ automatically if it does not exist.
DATA_DIR.mkdir(exist_ok=True)

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "chunks.json"

# Cheap and suitable embedding model for this project.
EMBEDDING_MODEL = "text-embedding-3-small"

# File: src/embedder.py
# Replace the existing create_embeddings() function with this version.

def create_embeddings(chunks, batch_size=50):
    """
    Generate embeddings in smaller batches.

    If OpenAI temporarily returns a rate-limit error,
    wait and retry instead of crashing the entire program.
    """

    all_embeddings = []

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        texts = [chunk["text"] for chunk in batch]

        print(
            f"Embedding chunks "
            f"{start + 1}-{end} "
            f"of {len(chunks)}"
        )

        while True:
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )

                batch_embeddings = [
                    item.embedding
                    for item in response.data
                ]

                all_embeddings.extend(batch_embeddings)

                # Small pause to avoid hitting the TPM rate limit.
                time.sleep(1)

                break

            except RateLimitError:
                print("Rate limit reached. Waiting 5 seconds...")
                time.sleep(5)

    return np.array(all_embeddings, dtype=np.float32)


if __name__ == "__main__":
    print("Loading PDFs...")
    documents = load_pdf_documents()

    print("\nCreating chunks...")
    chunks = chunk_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\nCreating embeddings...")
    embeddings = create_embeddings(chunks)

    # Save vectors.
    np.save(EMBEDDINGS_FILE, embeddings)

    # Save original chunks + metadata.
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nEmbedding generation complete.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embeddings: {EMBEDDINGS_FILE}")
    print(f"Saved chunks: {CHUNKS_FILE}")