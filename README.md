# pdf-rag-assistant-streamlit
# PDF RAG Assistant with Streamlit

A Streamlit-based AI assistant that answers questions from uploaded PDF documents using OpenAI embeddings, FAISS, and a fallback to general LLM knowledge.

## Overview

This project is a simple Retrieval-Augmented Generation (RAG) application built with Streamlit.  
It allows a user to upload a PDF, ask questions about its content, retrieve relevant passages, and generate an answer grounded in the document whenever possible.

If no sufficiently relevant passage is found, the assistant falls back to the model's general knowledge.

## Features

- Upload a PDF document from the interface
- Extract and split text into passages
- Generate embeddings with OpenAI
- Retrieve similar passages with FAISS
- Use only relevant passages below a distance threshold
- Fall back to general LLM knowledge when needed
- Display the source passages used for the answer
- Simple and interactive Streamlit interface

## Demo

![Application demo](assets/demo.gif)

## Tech Stack

- Python
- Streamlit
- PyPDF2
- LangChain
- OpenAI Embeddings / Chat Model
- FAISS

## Project Structure

```text
pdf-rag-assistant-streamlit/
├── AssistantIA.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
│   ├── demo.gif
│   └── interface.png
└── .streamlit/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nawelbenchaabane/pdf-rag-assistant-streamlit.git
cd pdf-rag-assistant-streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a file named `.streamlit/secrets.toml` and add your OpenAI API key:

```toml
OPENAI_API_KEY = "your_openai_api_key"
```

## Run the App

```bash
streamlit run AssistantIA.py
```

## How It Works

1. The user uploads a PDF document.
2. The app extracts the text from the PDF.
3. The text is split into smaller passages.
4. Embeddings are generated for each passage.
5. FAISS retrieves the top-k most similar passages for the question.
6. Only passages with a distance below the selected threshold are kept.
7. If at least one relevant passage is found, the model answers using the document context.
8. Otherwise, the assistant answers using general model knowledge.

## Notes

- A lower distance means a passage is more relevant to the question.
- The app uses a configurable distance threshold to decide whether to rely on the document.
- Only filtered relevant passages are sent to the model.
- Scanned PDFs or image-only PDFs may not be parsed correctly with basic text extraction.

## Possible Improvements

- OCR support for scanned PDFs
- Better document parsing
- Multi-file support
- Conversation memory across multiple questions
- Deployment on Streamlit Community Cloud
- More advanced retriever/reranker logic

## Author

Created by Nawel Benchaabane.
