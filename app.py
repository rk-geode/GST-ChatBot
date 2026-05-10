"""FastAPI application for GST compliance chatbot."""

import sys
import logging
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import create_rag_pipeline, GSTRAGPipeline
from src.helper import (
    validate_env_vars,
    format_response,
    get_env_var,
    get_project_root
)
from src.document_loader import load_gst_documents
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="GST Compliance Chatbot",
    description="RAG-powered chatbot for GST compliance questions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Global RAG pipeline instance
rag_pipeline: Optional[GSTRAGPipeline] = None


# Pydantic models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., min_length=1, max_length=1000)
    include_sources: bool = Field(default=True)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    question: str
    sources: Optional[List[dict]] = None
    num_sources: int = 0


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    message: str
    chunks_ingested: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    index_name: str
    total_vectors: int
    dimension: int


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global rag_pipeline
    logger.info("Starting GST Chatbot API...")

    try:
        if validate_env_vars():
            rag_pipeline = create_rag_pipeline()
            logger.info("RAG pipeline initialized successfully")
        else:
            logger.warning("RAG pipeline not initialized - missing environment variables")
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/api")
async def root():
    """Health check endpoint."""
    return format_response(
        data={
            "service": "GST Compliance Chatbot",
            "status": "running",
            "version": "1.0.0"
        },
        message="Welcome to GST Compliance Chatbot"
    )


@app.get("/health")
async def health_check():
    """Detailed health check."""
    env_ok = validate_env_vars()
    pipeline_ok = rag_pipeline is not None

    return format_response(
        data={
            "environment_variables": env_ok,
            "rag_pipeline": pipeline_ok,
            "overall": env_ok and pipeline_ok
        },
        message="Health check completed"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a GST compliance question.

    Args:
        request: Chat request with question

    Returns:
        Chat response with answer and sources
    """
    if not request.question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized. Check environment variables."
        )

    try:
        result = rag_pipeline.ask(request.question)

        return ChatResponse(
            answer=result["answer"],
            question=result["question"],
            sources=result.get("sources", []) if request.include_sources else None,
            num_sources=result.get("num_sources", 0)
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(data_dir: str = "data"):
    """Ingest documents into the vector store.

    Args:
        data_dir: Directory containing PDF documents

    Returns:
        Ingest response with number of chunks ingested
    """
    if not validate_env_vars():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Missing required environment variables"
        )

    try:
        project_root = get_project_root()
        docs_dir = project_root / data_dir

        logger.info(f"Ingesting documents from {docs_dir}")

        # Load documents
        documents = load_gst_documents(str(docs_dir))

        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No documents found in {data_dir}"
            )

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            model=get_env_var("EMBEDDING_MODEL", "text-embedding-ada-002"),
            openai_api_key=get_env_var("OPENAI_API_KEY")
        )

        # Get Pinecone index
        pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))
        index = pc.Index(get_env_var("PINECONE_INDEX_NAME", "gst-chatbot"))

        # Create vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )

        # Add documents
        vector_store.add_documents(documents)

        # Reset pipeline to reload from updated index
        global rag_pipeline
        rag_pipeline = create_rag_pipeline()

        logger.info(f"Successfully ingested {len(documents)} chunks")

        return IngestResponse(
            message="Documents ingested successfully",
            chunks_ingested=len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting documents: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get vector store statistics.

    Returns:
        Stats response with index information
    """
    if not validate_env_vars():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Missing required environment variables"
        )

    try:
        pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))
        index = pc.Index(get_env_var("PINECONE_INDEX_NAME", "gst-chatbot"))
        stats = index.describe_index_stats()

        return StatsResponse(
            index_name=get_env_var("PINECONE_INDEX_NAME", "gst-chatbot"),
            total_vectors=stats.total_vector_count,
            dimension=stats.dimension
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)