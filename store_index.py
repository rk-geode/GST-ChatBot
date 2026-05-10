"""Script to ingest GST documents into Pinecone vector store."""

import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import load_gst_documents
from src.helper import get_project_root, get_env_var, validate_env_vars
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def create_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int = 1536):
    """Create Pinecone index if it doesn't exist.

    Args:
        pc: Pinecone client
        index_name: Name of the index
        dimension: Embedding dimension
    """
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": get_env_var("PINECONE_ENVIRONMENT", "us-west1")
                }
            }
        )
        logger.info(f"Index '{index_name}' created successfully")
    else:
        logger.info(f"Index '{index_name}' already exists")


def ingest_documents(data_dir: str = "data"):
    """Ingest GST documents into Pinecone vector store.

    Args:
        data_dir: Directory containing PDF documents
    """
    # Validate environment variables
    if not validate_env_vars():
        raise ValueError("Missing required environment variables")

    # Initialize Pinecone
    pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))
    index_name = get_env_var("PINECONE_INDEX_NAME", "gst-chatbot")

    # Create index if needed
    create_index_if_not_exists(pc, index_name)

    # Load documents
    logger.info(f"Loading documents from {data_dir}")
    documents = load_gst_documents(data_dir)

    if not documents:
        logger.warning("No documents found to ingest")
        return

    logger.info(f"Loaded {len(documents)} document chunks")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=get_env_var("EMBEDDING_MODEL", "text-embedding-ada-002"),
        openai_api_key=get_env_var("OPENAI_API_KEY")
    )

    # Get Pinecone index
    index = pc.Index(index_name)

    # Create vector store and ingest
    vector_store = PineconeLangChain(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Add documents to vector store
    logger.info("Adding documents to Pinecone...")
    vector_store.add_documents(documents)

    logger.info(f"Successfully ingested {len(documents)} document chunks into '{index_name}'")

    # Get stats
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")


def main():
    """Main function to run the ingestion."""
    project_root = get_project_root()
    data_dir = project_root / "data"

    logger.info("Starting GST document ingestion...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")

    try:
        ingest_documents(str(data_dir))
        logger.info("Ingestion completed successfully!")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()