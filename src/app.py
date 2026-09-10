# File: src/app.py
# Purpose:
# Simple Streamlit UI for the Industrial Robotics RAG Assistant.

import streamlit as st

from answerer import answer_question


st.set_page_config(
    page_title="Industrial Robotics RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Industrial Robotics RAG Assistant")

st.write(
    """
    Ask questions about the industrial robot manuals stored in this RAG system.
    Answers are generated only from retrieved document context.
    """
)

question = st.text_input(
    "Ask a question",
    placeholder="Example: What is the maximum payload of the UR5e?",
)

if st.button("Ask") and question:

    with st.spinner("Searching robot manuals..."):

        answer, results = answer_question(question)

    # Show final grounded answer.
    st.subheader("Answer")
    st.markdown(answer)

    # Show retrieved evidence so the user can inspect the RAG process.
    st.subheader("Retrieved Sources")

    for index, result in enumerate(results, start=1):

        with st.expander(
            f"[S{index}] {result['source']} "
            f"— Page {result['page']} "
            f"— Similarity {result['score']:.4f}"
        ):
            st.write(result["text"])