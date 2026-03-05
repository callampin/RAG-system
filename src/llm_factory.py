import os
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from .config import config


class LLMFactory:
    _llm_instance: Optional[BaseChatModel] = None
    _embeddings_instance: Optional[Embeddings] = None

    @classmethod
    def get_llm(cls) -> BaseChatModel:
        if cls._llm_instance is not None:
            return cls._llm_instance

        provider = config.ACTIVE_LLM

        if provider == "gemini":
            cls._llm_instance = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=config.GEMINI_API_KEY,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                convert_system_message_to_human=True,
            )
        elif provider == "openai":
            cls._llm_instance = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=config.OPENAI_API_KEY,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
        elif provider == "minimax":
            cls._llm_instance = ChatOpenAI(
                model="MiniMax/MiniMax-M2.5",
                openai_api_key=config.MINIMAX_API_KEY,
                base_url="https://api.minimax.chat/v1",
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return cls._llm_instance

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        if cls._embeddings_instance is not None:
            return cls._embeddings_instance

        provider = config.ACTIVE_LLM

        if provider == "gemini":
            cls._embeddings_instance = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=config.GEMINI_API_KEY,
            )
        elif provider == "openai":
            cls._embeddings_instance = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=config.OPENAI_API_KEY,
            )
        elif provider == "minimax":
            cls._embeddings_instance = OpenAIEmbeddings(
                model="embo-01",
                openai_api_key=config.MINIMAX_API_KEY,
                base_url="https://api.minimax.chat/v1",
            )
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")

        return cls._embeddings_instance

    @classmethod
    def reset(cls):
        cls._llm_instance = None
        cls._embeddings_instance = None


def get_llm() -> BaseChatModel:
    return LLMFactory.get_llm()


def get_embeddings() -> Embeddings:
    return LLMFactory.get_embeddings()
