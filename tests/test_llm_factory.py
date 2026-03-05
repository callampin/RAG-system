import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_factory import LLMFactory, get_llm, get_embeddings
from src.config import config


class TestLLMFactory:
    def test_config_has_active_llm(self):
        assert config.ACTIVE_LLM in ["gemini", "openai", "minimax"]

    def test_llm_factory_imports(self):
        assert LLMFactory is not None
        assert get_llm is not None
        assert get_embeddings is not None

    def test_factory_reset(self):
        LLMFactory.reset()
        assert LLMFactory._llm_instance is None
        assert LLMFactory._embeddings_instance is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
