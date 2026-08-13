[README (1).md](https://github.com/user-attachments/files/31022622/README.1.md)
# RAG Chatbot with Groq & LangChain

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, and **Groq's LLaMA 3** model. The app answers user questions based on the content of a PDF document, using semantic search to retrieve relevant context before generating a response.

## Overview

This project loads a PDF, splits it into chunks, embeds those chunks into a vector store, and uses that store to retrieve relevant context for each user question. The retrieved context is passed to a Groq-hosted LLaMA 3 model, which generates an accurate, grounded answer — all through a simple chat interface.

## Features

- 💬 Interactive chat interface built with Streamlit
- 📄 Answers grounded in the content of a source PDF (no hallucinated info)
- 🔍 Semantic search over document chunks using HuggingFace embeddings
- ⚡ Fast inference powered by Groq's LLaMA 3 (8B) model
- 🧠 Persistent chat history within a session
- 🗂️ Automatic chunking and indexing of source documents

## Tech Stack

| Component        | Technology                                |
|-------------------|--------------------------------------------|
| UI                | [Streamlit](https://streamlit.io/)         |
| LLM Orchestration | [LangChain](https://www.langchain.com/)    |
| LLM Provider      | [Groq](https://groq.com/) (LLaMA 3 - 8B)   |
| Embeddings        | HuggingFace (`all-MiniLM-L12-v2`)          |
| Vector Store      | Chroma (via `VectorstoreIndexCreator`)     |
| Document Loader   | `PyPDFLoader`                              |
| Config            | `python-dotenv`                            |

## Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys)

### Installation

1. Clone the repository
   ```bash
    git clone https://github.com/saqibmasoodai-ops/Rag-Appilcation.git
    cd Rag-Appilcation
   ```

2. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install streamlit langchain langchain-groq langchain-community \
               pypdf chromadb sentence-transformers python-dotenv
   ```

### Configuration

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit your `.env` file or hardcode your API key in the source code. Add `.env` to your `.gitignore`.

### Add Your Document

Place the PDF you want the chatbot to answer questions from in the project root, and update the `pdf_name` variable in the code to match your file name:

```python
pdf_name = "your-document.pdf"
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Launch the app.
2. Type a question related to the content of your PDF in the chat input box.
3. The chatbot retrieves the most relevant sections of the document and generates a precise answer using Groq's LLaMA 3 model.
4. Chat history persists for the duration of your session.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── .env                     # API keys (not committed)
├── your-document.pdf        # Source PDF for retrieval
└── README.md
```

## How It Works

1. **Load** — The PDF is loaded using `PyPDFLoader`.
2. **Split** — Text is split into overlapping chunks (1000 characters, 100 overlap) via `RecursiveCharacterTextSplitter`.
3. **Embed & Index** — Each chunk is embedded using a HuggingFace sentence-transformer model and stored in a Chroma vector store.
4. **Retrieve** — On each query, the top 3 most relevant chunks are retrieved.
5. **Generate** — The retrieved context and user question are passed to Groq's LLaMA 3 model via a `RetrievalQA` chain to produce the final answer.

## Known Limitations

- The vector store is cached in memory per session (`@st.cache_resource`); it rebuilds if the app restarts.
- Only a single, hardcoded PDF is supported per run — no file upload UI yet.
- No conversation-aware retrieval (each query is treated independently, not as part of an ongoing dialogue).

## License

This project is licensed under the [MIT License](LICENSE).
