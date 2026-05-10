"""Helper utilities for GST chatbot."""

import os
import logging
from typing import Optional, Any
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set")
    return value


def validate_env_vars() -> bool:
    """Validate that required environment variables are set.

    Returns:
        True if all required variables are set
    """
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME"
    ]

    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False

    logger.info("All required environment variables are set")
    return True


def format_response(data: dict, success: bool = True, message: str = "") -> dict:
    """Format API response.

    Args:
        data: Response data
        success: Whether the operation was successful
        message: Optional message

    Returns:
        Formatted response dictionary
    """
    return {
        "success": success,
        "message": message,
        "data": data
    }


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def ensure_data_directory() -> Path:
    """Ensure the data directory exists.

    Returns:
        Path to data directory
    """
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


class GSTChatError(Exception):
    """Base exception for GST chatbot errors."""

    pass


class DocumentProcessingError(GSTChatError):
    """Error during document processing."""

    pass


class VectorStoreError(GSTChatError):
    """Error with vector store operations"""

    pass


class LLMError(GSTChatError):
    """Error with LLM operations"""

    pass