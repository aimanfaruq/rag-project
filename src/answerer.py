# File: src/answerer.py
# Purpose:
# Retrieve relevant chunks and ask an LLM to answer
# using ONLY those retrieved chunks.

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from search import search


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load API key from .env
load_dotenv(PROJECT_ROOT / ".env")

client = OpenAI()


def build_context(results):
    """
    Convert retrieved chunks into labeled source sections.

    The labels [S1], [S2], etc. let the LLM cite
    which retrieved chunk supports its answer.
    """

    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"""
[S{index}]
Source: {result["source"]}
Page: {result["page"]}
Similarity score: {result["score"]:.4f}

{result["text"]}
""".strip()
        )

    return "\n\n".join(context_parts)


def answer_question(question, top_k=5):
    """
    Full RAG flow:

    Question
        ↓
    Retrieve relevant chunks
        ↓
    Build context
        ↓
    LLM generates grounded answer
    """

    results = search(question, top_k=top_k)

    context = build_context(results)

    instructions = """
You are a technical assistant for industrial robotics documentation.

Answer the user's question using ONLY the provided retrieved context.

Rules:
- Do not use outside knowledge.
- If the retrieved context is insufficient, say so.
- Cite supporting information using [S1], [S2], etc.
- Prefer concise, factual answers.
""".strip()

    prompt = f"""
USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}
""".strip()

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=instructions,
        input=prompt,
    )

    return response.output_text, results


if __name__ == "__main__":
    question = input("\nAsk the robot documentation: ")

    answer, results = answer_question(question)

    print("\n--- RAG ANSWER ---")
    print(answer)

    print("\n--- SOURCES USED ---")

    for index, result in enumerate(results, start=1):
        print(
            f"[S{index}] "
            f"{result['source']} "
            f"- page {result['page']} "
            f"- score {result['score']:.4f}"
        )