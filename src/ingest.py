#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.llm_factory import get_embeddings
from langchain_chroma import Chroma


def ingest_documents(data_path: str = None, persist_dir: str = None):
    data_path = data_path or config.DATA_PATH
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIR

    print(f"\n{'='*60}")
    print("RAG INGESTION PIPELINE")
    print(f"{'='*60}")
    print(f"Data path: {data_path}")
    print(f"Persist dir: {persist_dir}")
    print(f"Chunk size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}")
    print(f"Active LLM: {config.ACTIVE_LLM}")
    print(f"{'='*60}\n")

    config.validate()

    print("[1/4] Loading documents...")
    loader = DocumentLoader(data_path)
    documents = loader.load()

    if not documents:
        print("No documents found in the data path!")
        return

    print(f"[2/4] Splitting documents into chunks...")
    text_splitter = TextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)

    print("[3/4] Creating embeddings and vectorstore...")
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"[4/4] Vectorstore persisted to: {persist_dir}")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"\n✅ Ingestion complete!")
    print(f"\nYou can now run the app with: streamlit run app/main.py")
    print(f"Or with Docker: docker-compose up")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to directory containing PDFs",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist ChromaDB",
    )
    args = parser.parse_args()

    try:
        ingest_documents(args.data_path, args.persist_dir)
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
