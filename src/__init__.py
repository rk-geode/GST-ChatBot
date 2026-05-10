"""GST Chatbot package."""

from .document_loader import GSTDocumentLoader, load_gst_documents
from .rag_pipeline import GSTRAGPipeline, create_rag_pipeline
from .prompt import GST_SYSTEM_PROMPT, GST_QUESTION_PROMPT
from .helper import (
    get_env_var,
    validate_env_vars,
    format_response,
    get_project_root,
    GSTChatError,
    DocumentProcessingError,
    VectorStoreError,
    LLMError
)

__version__ = "1.0.0"
__all__ = [
    "GSTDocumentLoader",
    "load_gst_documents",
    "GSTRAGPipeline",
    "create_rag_pipeline",
    "GST_SYSTEM_PROMPT",
    "GST_QUESTION_PROMPT",
    "get_env_var",
    "validate_env_vars",
    "format_response",
    "get_project_root",
    "GSTChatError",
    "DocumentProcessingError",
    "VectorStoreError",
    "LLMError"
]