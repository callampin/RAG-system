import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

load_dotenv(BASE_DIR / ".env")


class Config:
    ACTIVE_LLM: str = os.getenv("ACTIVE_LLM", "gemini").lower()
    
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MINIMAX_API_KEY: Optional[str] = os.getenv("MINIMAX_API_KEY")
    MINIMAX_GROUP_ID: Optional[str] = os.getenv("MINIMAX_GROUP_ID")
    
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "vectorstore"))
    DATA_PATH: str = os.getenv("DATA_PATH", str(BASE_DIR / "data" / "pdfs"))
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    
    @classmethod
    def validate(cls) -> bool:
        if cls.ACTIVE_LLM == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when ACTIVE_LLM=gemini")
        if cls.ACTIVE_LLM == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when ACTIVE_LLM=openai")
        if cls.ACTIVE_LLM == "minimax" and not (cls.MINIMAX_API_KEY and cls.MINIMAX_GROUP_ID):
            raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID are required when ACTIVE_LLM=minimax")
        return True


config = Config()
