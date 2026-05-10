"""Document loader and text processing for GST documents."""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class GSTDocumentLoader:
    """Loader for GST PDF documents."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the document loader.

        Args:
            data_dir: Directory containing PDF documents
        """
        self.data_dir = Path(data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load a single PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of document chunks
        """
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        # Add source metadata to each page
        for page in pages:
            page.metadata["source"] = file_path.name
            page.metadata["file_type"] = "pdf"

        logger.info(f"Loaded {len(pages)} pages from {file_path.name}")
        return pages

    def load_all_pdfs(self) -> List[Document]:
        """Load all PDF files from the data directory.

        Returns:
            List of all loaded documents
        """
        all_documents = []

        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return all_documents

        pdf_files = list(self.data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                documents = self.load_pdf(pdf_file)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")

        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process_directory(self) -> List[Document]:
        """Load and chunk all documents from the data directory.

        Returns:
            List of processed document chunks
        """
        documents = self.load_all_pdfs()
        chunks = self.chunk_documents(documents)
        return chunks


def load_gst_documents(data_dir: str = "data") -> List[Document]:
    """Convenience function to load all GST documents.

    Args:
        data_dir: Directory containing PDF documents

    Returns:
        List of processed document chunks
    """
    loader = GSTDocumentLoader(data_dir)
    return loader.process_directory()