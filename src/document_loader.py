from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

    def load(self) -> List[Document]:
        loader = PyPDFDirectoryLoader(
            str(self.data_path),
            extract_images=True,
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {self.data_path}")
        return documents

    def load_file(self, file_path: str) -> List[Document]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()
