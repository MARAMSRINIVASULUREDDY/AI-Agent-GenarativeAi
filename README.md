# AI PDF Chatbot (Generative AI Project)

**Overview**
This project is a PDF-based AI chatbot that allows users to upload a document and ask questions.  
It extracts content, processes it, and generates summarized answers using a local LLM.

**Features**
- Upload and read PDF documents
- Intelligent text chunking
- Semantic search using FAISS
- Context-based answer generation
- Clean bullet-point summaries
- Fully free (no paid APIs)

**Key concepts used**

- Retrieval-Augmented Generation (RAG)
- Embeddings & Vector Search (FAISS)
- LLM integration using HuggingFace
- Streamlit for UI

**Tech Stack**
- Python
- Streamlit (UI)
- LangChain
- FAISS (Vector Database)
- HuggingFace Transformers
- Sentence Transformers (Embeddings)

**How it Works**
1. Extracts text from PDF
2. Splits text into chunks
3. Converts chunks into embeddings
4. Stores in FAISS vector database
5. Retrieves relevant context based on query
6. Generates response using GPT2 model

**Use Case**

This project demonstrates how AI agents can be used in real-world scenarios.

**Future Improvements**
- Add UI
- Improve accuracy
- Add more tools/integrations


**How to Run**
```bash
pip install -r requirements.txt
streamlit run main.py
