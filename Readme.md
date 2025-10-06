# llamaRAG

## Overview

llamaRAG is a Retrieval-Augmented Generation (RAG) system that provides two main functionalities:

1. **Confluence Documentation Q&A**: Query and get answers from Confluence wiki pages using natural language
2. **Java Repository Analysis**: Analyze Java codebases to extract database entities and their relationships

The system uses Ollama for local LLM inference, ChromaDB for vector storage, and LangChain for orchestration.

## Features

- **Confluence RAG**: Fetches, embeds, and enables Q&A on Confluence documentation
- **Repository RAG**: Analyzes Java repositories to identify database entities, tables, and columns
- **Vector-based search**: Uses embeddings for semantic search across documentation
- **Local LLM**: Runs entirely locally using Ollama (no external API calls needed)
- **Persistent storage**: ChromaDB stores embeddings for fast retrieval

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Git

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/dkovacevic/llamaRAG.git
   cd llamaRAG
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull required Ollama models:
   ```bash
   ollama pull mxbai-embed-large
   ollama pull gemma3
   ```

## Configuration

### Confluence RAG Setup

Set the following environment variables for Confluence access:

```bash
export CONFLUENCE_BASE_URL="https://your-confluence-instance.com"
export CONFLUENCE_USERNAME="your-username"
export CONFLUENCE_API_TOKEN="your-api-token"
```

Edit `confluence.py` to set your space key and parent page ID:
```python
SPACE_KEY = "ERPCRM"  # Your Confluence space key
PARENT_ID = "169651392"  # Parent page ID to start fetching from
```

### Repository RAG Setup

Edit `repoRAG.py` to configure the repository to analyze:
```python
REPO_URL = "https://your-repo-url.git"
LOCAL_PATH = "repo"
```

## Usage

### Confluence Documentation Q&A

Run the Confluence RAG system to ask questions about your documentation:

```bash
python3 main.py
```

On first run, it will:
1. Fetch all pages from the configured Confluence space
2. Create embeddings using `mxbai-embed-large`
3. Store them in ChromaDB (`./chrome_langchain_db/`)

Then you can ask questions in the interactive prompt:
```
question: How do I configure the authentication system?
```

Type `q` to quit.

### Java Repository Analysis

Run the repository analyzer to extract database entities:

```bash
python3 repoRAG.py
```

This will:
1. Clone or update the configured Git repository
2. Find all Java files
3. Create embeddings and analyze the code
4. Generate a report (`inter_service_report.json`) with identified database tables and columns

## Project Structure

```
llamaRAG/
├── main.py              # Confluence Q&A interactive interface
├── confluence.py        # Confluence API integration and page fetching
├── vector.py            # Vector store setup and document embedding
├── repoRAG.py           # Java repository analysis tool
├── requirements.txt     # Python dependencies
├── Readme.md           # This file
├── .gitignore          # Git ignore rules
├── chrome_langchain_db/ # ChromaDB vector store (created on first run)
├── pages/              # Downloaded Confluence pages (created on first run)
├── repo/               # Cloned repository (created by repoRAG.py)
└── chroma_db/          # Vector store for repo analysis (created by repoRAG.py)
```

## How It Works

### Confluence RAG Pipeline

1. **Data Ingestion** (`confluence.py`):
   - Fetches pages from Confluence using REST API
   - Extracts text content from HTML storage format
   - Saves pages locally for reference

2. **Embedding & Storage** (`vector.py`):
   - Splits documents into chunks (2000 chars with 200 char overlap)
   - Creates embeddings using Ollama's `mxbai-embed-large` model
   - Stores in ChromaDB for efficient similarity search

3. **Query & Retrieval** (`main.py`):
   - Takes user questions via interactive prompt
   - Retrieves top 5 most relevant document chunks
   - Generates answers using `gemma3` LLM
   - Citations show source document titles

### Repository RAG Pipeline

1. **Code Ingestion** (`repoRAG.py`):
   - Clones/updates Git repository
   - Finds all Java files recursively
   - Loads file contents as documents

2. **Analysis**:
   - Splits code into chunks for processing
   - Uses RAG to identify Spring Boot entities
   - Extracts `@Entity`, `@Table`, `@Column`, `@Id` annotations
   - Generates structured JSON report with tables and columns

## Dependencies

Core dependencies (see `requirements.txt`):
- `langchain` - LLM orchestration framework
- `langchain-ollama` - Ollama integration for LangChain
- `langchain-chroma` - ChromaDB vector store integration
- `langchain-community` - Community integrations
- `beautifulsoup4` / `bs4` - HTML parsing for Confluence pages
- `requests` - HTTP client for API calls

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT