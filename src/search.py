# File: src/search.py
# Purpose:
# Convert a user's question into an embedding,
# compare it with all document embeddings,
# and return the most relevant chunks.

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load API key from the project's .env file.
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI()

DATA_DIR = PROJECT_ROOT / "data"

EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
CHUNKS_FILE = DATA_DIR / "chunks.json"

EMBEDDING_MODEL = "text-embedding-3-small"


def load_index():
    """
    Load previously generated embeddings and chunk metadata.
    """

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return embeddings, chunks


def create_query_embedding(question):
    """
    Convert the user's question into the same vector space
    used for the document chunks.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    return np.array(
        response.data[0].embedding,
        dtype=np.float32,
    )


def cosine_similarity(query_vector, document_vectors):
    """
    Calculate cosine similarity between the query
    and every document chunk.
    """

    query_norm = np.linalg.norm(query_vector)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1,
    )

    similarities = (
        document_vectors @ query_vector
    ) / (document_norms * query_norm)

    return similarities


def search(question, top_k=5):
    """
    Return the top-k most relevant document chunks.
    """

    embeddings, chunks = load_index()

    query_embedding = create_query_embedding(question)

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )

    # Sort scores from highest to lowest.
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        chunk = chunks[index].copy()

        chunk["score"] = float(similarities[index])

        results.append(chunk)

    return results


if __name__ == "__main__":

    question = input("\nAsk a question: ")

    results = search(question)

    print("\n--- TOP RESULTS ---")

    for rank, result in enumerate(results, start=1):

        print(f"\n===== RESULT {rank} =====")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk: {result['chunk']}")

        print("\nText:")
        print(result["text"][:1200])