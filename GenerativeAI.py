import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# UI
st.header("AI PDF Chatbot (Free)")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:

    # Read PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)

    # Embeddings 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Ask something from PDF:")

    if user_question:

        docs = vector_store.similarity_search(user_question, k=2)
        context = " ".join([doc.page_content for doc in docs])

        # LIMIT context (important)
        context = context[:1500]

        # FREE MODEL (STABLE)
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            max_new_tokens=150
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # SIMPLE PROMPT (no confusion)
        prompt = f"""
        Give a short summary in 5-10 bullet points:

        {context}
        """

        response = llm.invoke(prompt)

        result = str(response)

        #  Clean text
        result = result.replace("\n", " ")
        result = result.replace("•", "-")

        # Split into sentences and make bullets
        points = result.split(". ")

        st.write("Answer:")

        clean_points = []

        for p in points:
            p = p.strip()

            # remove unwanted lines
            if (
                    len(p) < 30 or
                    "Figure" in p or
                    "Customer Preferences" in p or
                    p.lower().startswith("give a short")
            ):
                continue

            clean_points.append(p)

        # show only top 4–5 clean points
        for p in clean_points[:5]:
            st.write(f"- {p}")
