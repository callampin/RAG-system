from pathlib import Path
from typing import Optional, Dict, Any, List
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import config
from src.llm_factory import get_embeddings
from src.chain_builder import build_qa_chain


class RAGEngine:
    def __init__(self, persist_dir: str = None, collection_name: str = "rag_docs"):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self._vectorstore: Optional[Chroma] = None
        self._qa_chain: Optional[Any] = None

    def load_vectorstore(self) -> Chroma:
        if self._vectorstore is not None:
            return self._vectorstore

        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.persist_dir}. "
                f"Run ingestion first: python -m src.ingest"
            )

        embeddings = get_embeddings()
        self._vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
            collection_name=self.collection_name,
        )

        print(f"Loaded vectorstore from: {self.persist_dir}")
        return self._vectorstore

    def get_retriever(self, k: int = 4):
        vectorstore = self.load_vectorstore()
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def initialize_qa_chain(self, k: int = 4):
        retriever = self.get_retriever(k=k)
        self._qa_chain = build_qa_chain(retriever)
        return self._qa_chain

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        if self._qa_chain is None:
            self.initialize_qa_chain(k=k)

        result = self._qa_chain.invoke({"query": question})

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        if not source_docs:
            answer = (
                "No tengo información suficiente en la documentación para responder esta pregunta. "
                "Por favor, contacta con soporte técnico humano."
            )

        return {
            "answer": answer,
            "sources": [self._format_source(doc) for doc in source_docs],
            "num_sources": len(source_docs),
        }

    def query_with_context(self, question: str, k: int = 4) -> Dict[str, Any]:
        vectorstore = self.load_vectorstore()
        retriever = self.get_retriever(k=k)
        
        docs = retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "No tengo información suficiente en la documentación para responder esta pregunta.",
                "context": [],
                "sources": [],
            }

        context = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "answer": "",
            "context": context,
            "sources": [self._format_source(doc) for doc in docs],
            "num_sources": len(docs),
        }

    def _format_source(self, doc: Document) -> Dict[str, str]:
        source = doc.metadata.get("source", "Unknown")
        if isinstance(source, str):
            source = Path(source).name
        return {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": source,
            "page": doc.metadata.get("page", "N/A"),
        }

    def get_stats(self) -> Dict[str, Any]:
        vectorstore = self.load_vectorstore()
        count = vectorstore._collection.count()
        return {
            "total_documents": count,
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
        }

    def reset(self):
        self._vectorstore = None
        self._qa_chain = None


rag_engine = RAGEngine()
