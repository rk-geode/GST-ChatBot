# GST Compliance Chatbot

A RAG-powered chatbot that answers Goods and Services Tax (GST) compliance questions for India using official circulars and notifications.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastAPI   │────▶│   LangChain  │────▶│   OpenAI    │
│   Backend   │     │   RAG Pipe   │     │   LLM       │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   Pinecone   │
                    │ Vector Store │
                    └──────────────┘
```

## Tech Stack

- **Backend**: FastAPI
- **LLM**: OpenAI GPT (gpt-3.5-turbo)
- **Vector Database**: Pinecone
- **Framework**: LangChain
- **Document Processing**: PyPDF

## Features

- RAG-based question answering over GST documents
- Accurate answers citing source documents
- RESTful API with FastAPI
- Document ingestion pipeline
- Source attribution for responses

## Installation

1. Clone the repository and navigate to the project directory:

```bash
cd GST-ChatBot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create environment file:

```bash
copy .env.example .env
```

4. Edit `.env` with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-west1
PINECONE_INDEX_NAME=gst-chatbot
EMBEDDING_MODEL=text-embedding-ada-002
CHAT_MODEL=gpt-3.5-turbo
```

## Usage

### 1. Add GST Documents

Place your PDF documents (circulars, notifications) in the `data/` folder.

### 2. Ingest Documents

```bash
python store_index.py
```

This will:
- Load all PDFs from `data/`
- Split into chunks (1000 chars, 200 overlap)
- Create embeddings using OpenAI
- Store in Pinecone vector database

### 3. Start the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### 4. Test the Chatbot

Using curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the due date for GSTR-3B filing?"}'
```

Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "What is the due date for GSTR-3B filing?"}
)
print(response.json())
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed status |
| `/chat` | POST | Ask a GST question |
| `/ingest` | POST | Re-ingest documents |
| `/stats` | GET | Vector store statistics |

### Example Response

```json
{
  "answer": "The due date for GSTR-3B filing is the 20th of the following month...",
  "question": "What is the due date for GSTR-3B filing?",
  "sources": [
    {
      "content": "GSTR-3B return shall be filed monthly...",
      "source": "gstr3Bofflineutility.pdf",
      "page": 1
    }
  ],
  "num_sources": 1
}
```

## Project Structure

```
GST-ChatBot/
├── .env.example          # Environment template
├── requirements.txt      # Dependencies
├── app.py                # FastAPI application
├── store_index.py        # Document indexing script
├── src/
│   ├── __init__.py       # Package exports
│   ├── document_loader.py # PDF loading & chunking
│   ├── rag_pipeline.py   # RAG chain
│   ├── prompt.py         # Prompt templates
│   └── helper.py         # Utilities
├── data/                 # PDF documents
│   └── *.pdf
└── templates/            # Frontend templates (optional)
```

## GST Topics Supported

- GST registration requirements
- Input Tax Credit (ITC)
- GSTR filing (GSTR-1, GSTR-2, GSTR-3B, GSTR-9)
- E-way bill generation
- GST rates and exemptions
- Reverse Charge Mechanism (RCM)
- Composition scheme
- TDS and TCS under GST
- GST refund procedures
- Export and import under GST

## License

MIT License