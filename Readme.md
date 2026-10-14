# DocuMind RAG 

An AI-powered PDF question-answering application built with Retrieval-Augmented Generation (RAG). Upload PDF documents and ask questions in natural language to receive context-grounded answers.

## Features

- PDF upload and text extraction
- Text chunking and semantic embeddings
- ChromaDB vector storage
- MMR-based document retrieval
- Groq-powered LLM responses
- Multiple answer styles
- Source chunks with page references
- Interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Groq API

## Run Locally

1. Clone this repository.
2. Create a virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Create a `.env` file and configure `GROQ_API_KEY`.
5. Run `streamlit run app.py`.

## Environment Variables

`GROQ_API_KEY` — Your Groq API key. Never commit your API key to GitHub.

## Future Improvements

- Deploy a public demo
- Support OCR for scanned PDFs
- Evaluate retrieval quality with test questions
