import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import RAGEngine


class TestRAGEngine:
    def test_rag_engine_init(self):
        engine = RAGEngine()
        assert engine.persist_dir is not None
        assert engine.collection_name == "rag_docs"

    def test_rag_engine_custom_params(self):
        engine = RAGEngine(persist_dir="/tmp/test", collection_name="test")
        assert engine.persist_dir == "/tmp/test"
        assert engine.collection_name == "test"

    def test_rag_engine_reset(self):
        engine = RAGEngine()
        engine.reset()
        assert engine._vectorstore is None
        assert engine._qa_chain is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
