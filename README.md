# RAG Personalizado para Atención al Cliente

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker" alt="Docker Ready">
  <a href="README-ESP.md"><img src="https://img.shields.io/badge/Español-Spanish-red" alt="Español"></a>
</p>

> *"Your AI-powered 24/7 technical support agent — answering customer questions ONLY from your documentation, eliminating hallucinations and building trust."*

---

## The Problem

Traditional customer support faces three critical challenges:

1. **Repetitive Questions** — Support teams spend 60%+ of their time answering the same questions over and over.

2. **High Support Costs** — 24/7 human support is expensive and difficult to scale during peak times.

3. **AI Hallucinations** — Generic chatbots invent dangerous answers that damage brand trust and can lead to liability issues.

---

## How It Works

| Step | Description |
|------|-------------|
| **1. Ingestion** | PDFs are loaded and split into overlapping chunks (1000 chars / 200 overlap) for context preservation |
| **2. Vectorization** | Each chunk is converted to an embedding vector using the same LLM provider's embedding model |
| **3. Semantic Search** | User queries are matched against the vector database to retrieve the most relevant context |
| **4. Constrained Generation** | The LLM (temperature=0.0) generates answers ONLY from retrieved context — it cannot hallucinate |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG CUSTOMER SUPPORT SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
   │  PDF Files  │            │   Docker    │            │   .env      │
   │ (docs/manuals)           │  Container  │            │ (API Keys)  │
   └─────────────┘            └─────────────┘            └─────────────┘
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  PyPDF      │                    │                           │
   │  Loader     │                    │                           │
   └─────────────┘                    │                           │
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  Text       │                    │                           │
   │  Splitter   │                    │                           │
   │ (1000/200)  │                    │                           │
   └─────────────┘                    │                           │
          │                           │                           │
          ▼                           │                           │
   ┌─────────────┐                    │                           │
   │  ChromaDB   │◄───────────────────┘                           │
   │  Vector     │                                                │
   │  Store      │                                                │
   └─────────────┘                                                │
          │                                                       │
          ▼                                                       │
   ┌─────────────────────────────────────────────────────────────┐
   │                      LLM FACTORY                            │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
   │  │   Gemini    │  │   OpenAI    │  │  MiniMax    │          │
   │  │   3 Flash   │  │ GPT-4o Mini │  │    M2.5     │          │
   │  └─────────────┘  └─────────────┘  └─────────────┘          │
   │                         │                                   │
   │                    get_llm()                                │
   │                    get_embeddings()                         │
   └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                    RAG ENGINE                               │
   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
   │  │  Retrieval  │───►│   Prompt    │───►│     LLM     │      │
   │  │  (ChromaDB) │    │  Template   │    │ (temp=0.0)  │      │
   │  └─────────────┘    └─────────────┘    └─────────────┘      │
   │                                                   │         │
   │                              ┌────────────────────┘         │
   │                              ▼                              │
   │                     ┌─────────────────┐                     │
   │                     │  "I don't know" │                     │
   │                     │  if no context  │                     │
   │                     └─────────────────┘                     │
   └─────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      STREAMLIT UI                           │
   │  ┌─────────────────────────────────────────────────────┐    │
   │  │  💬 Chat Interface    │  📄 Sources    │ ⚙️ Config  │    │
   │  └─────────────────────────────────────────────────────┘    │
   └─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core language |
| **LangChain** | RAG orchestration & LLM abstraction |
| **ChromaDB** | Local persistent vector store |
| **Streamlit** | Web UI / Chat interface |
| **Docker** | Containerized deployment |

---

## Key Features

- **🤖 Multi-LLM Agnostic** — Switch between Gemini 3, OpenAI GPT-4o, and MiniMax M2.5 by changing one environment variable
- **🛡️ Zero Hallucination** — Temperature set to 0.0; the system only answers from ingested documents or admits it doesn't know
- **💾 Local Vector Store** — ChromaDB persists embeddings locally; no external cloud services required
- **📄 PDF Support** — Load technical documentation, manuals, and guides as PDF files
- **🔄 Context Overlap** — 1000-char chunks with 200-char overlap preserve context across boundaries
- **🐳 Docker Ready** — Full containerization with docker-compose for easy deployment

---

## Installation & Configuration

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG-system
```

### 2. Configure environment variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your preferred LLM provider:

```env
# LLM Configuration
ACTIVE_LLM=gemini          # Options: gemini, openai, minimax

# API Keys - Replace with your actual keys
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=
MINIMAX_API_KEY=
MINIMAX_GROUP_ID=

# Paths
CHROMA_PERSIST_DIR=./vectorstore
DATA_PATH=./data/pdfs

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TEMPERATURE=0.0
MAX_TOKENS=2000
```

### 3. Place your PDF documents

Add your technical documentation, manuals, or support guides to:

```
data/pdfs/
```

---

## Docker Deployment

### Step 1: Run the ingestion pipeline

This processes your PDFs and creates the vector store:

```bash
docker-compose --profile ingest run rag-ingest
```

### Step 2: Start the application

```bash
docker-compose up
```

### Step 3: Access the UI

Open your browser and navigate to:

```
http://localhost:8501
```

---

## Project Structure

```
RAG-system/
├── app/
│   ├── __init__.py
│   └── main.py              # Streamlit chat interface
├── src/
│   ├── __init__.py
│   ├── config.py            # Environment variable loader
│   ├── llm_factory.py       # LLM/Embeddings factory (agnostic)
│   ├── document_loader.py   # PyPDF loader
│   ├── text_splitter.py      # Chunking (1000/200 overlap)
│   ├── ingest.py            # Ingestion script (CLI)
│   ├── chain_builder.py     # RetrievalQA chain builder
│   └── rag_engine.py        # Main RAG engine
├── data/
│   └── pdfs/                # Source PDF documents
├── vectorstore/             # ChromaDB persistent storage
├── tests/
│   ├── test_llm_factory.py
│   └── test_rag_engine.py
├── .env.example            # Environment template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Disclaimer

> ⚠️ **This is a portfolio project.** This system is designed for demonstration and educational purposes. Before production deployment, consider:
> - Implementing proper authentication and authorization
> - Adding rate limiting and monitoring
> - Securing API keys and environment variables
> - Scaling the vector store for large document collections

---

<p align="center">
  <strong>Built with ❤️ using LangChain + ChromaDB + Streamlit</strong>
</p>
