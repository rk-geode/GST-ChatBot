"""RAG pipeline for GST chatbot."""

import logging
from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .prompt import GST_SYSTEM_PROMPT, GST_QUESTION_PROMPT
from .helper import get_env_var

logger = logging.getLogger(__name__)


class GSTRAGPipeline:
    """RAG pipeline for GST compliance questions."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chat_model: Optional[str] = None
    ):
        """Initialize the RAG pipeline.

        Args:
            index_name: Pinecone index name
            embedding_model: OpenAI embedding model
            chat_model: OpenAI chat model
        """
        self.index_name = index_name or get_env_var("PINECONE_INDEX_NAME", "gst-chatbot")
        self.embedding_model = embedding_model or get_env_var("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.chat_model = chat_model or get_env_var("CHAT_MODEL", "gpt-3.5-turbo")

        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None

    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings."""
        if self.embeddings is None:
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=get_env_var("OPENAI_API_KEY")
            )

    def connect_to_vectorstore(self):
        """Connect to existing Pinecone vector store."""
        self._initialize_embeddings()

        from pinecone import Pinecone

        pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))
        index = pc.Index(self.index_name)

        # Create a simple vector store wrapper
        from langchain.vectorstores import Pinecone as PineconeLangChain

        self.vector_store = PineconeLangChain(
            index=index,
            embedding=self.embeddings,
            text_key="text"
        )
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def _create_qa_chain(self):
        """Create the question-answering chain."""
        if self.vector_store is None:
            self.connect_to_vectorstore()

        # Create custom prompt
        prompt = PromptTemplate(
            template=GST_QUESTION_PROMPT,
            input_variables=["context", "question"]
        )

        # Initialize chat model
        llm = ChatOpenAI(
            model=self.chat_model,
            temperature=0.3,
            openai_api_key=get_env_var("OPENAI_API_KEY")
        )

        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a GST compliance question.

        Args:
            question: User's question about GST

        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            self._create_qa_chain()

        logger.info(f"Processing question: {question}")

        result = self.qa_chain({"query": question})

        # Format source documents
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })

        return {
            "question": question,
            "answer": result["result"],
            "sources": sources,
            "num_sources": len(sources)
        }


def create_rag_pipeline() -> GSTRAGPipeline:
    """Create and return a configured RAG pipeline instance."""
    return GSTRAGPipeline()