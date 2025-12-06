# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot system** that answers questions about course materials using semantic vector search (ChromaDB) and AI generation (Claude API with tool-based agent pattern).

## Core Architecture

### Three-Tier Structure

```
frontend/          → HTML/CSS/JS web interface
backend/           → Python FastAPI application
docs/              → Course material storage (txt/pdf/docx)
```

### Request Flow (Tool-Based Agent Pattern)

The system uses a **two-phase Claude API interaction** for intelligent search:

1. **User Query** → Frontend → FastAPI `/api/query` endpoint
2. **RAGSystem** orchestrates: SessionManager (history) → AIGenerator → Claude API
3. **Claude decides** whether to use `search_course_content` tool (based on query analysis)
4. **If tool needed**: ToolManager → CourseSearchTool → VectorStore → ChromaDB
   - Two-step vector search: (a) Resolve course name semantically, (b) Search content with filters
5. **Tool results** sent back to Claude → Claude synthesizes final answer
6. **Response** returned with source attribution → Frontend displays with markdown rendering

**Key insight**: The system makes **2 Claude API calls per query** (tool decision + synthesis), not traditional RAG with pre-search.

### Backend Component Responsibilities

- **`app.py`**: FastAPI endpoints (`/api/query`, `/api/courses`), serves frontend static files
- **`rag_system.py`**: Orchestrator coordinating all components, owns the query workflow
- **`ai_generator.py`**: Claude API integration with tool execution loop (handles `tool_use` stop reason)
- **`search_tools.py`**: Tool definitions and ToolManager registry (extensible Tool base class)
- **`vector_store.py`**: ChromaDB wrapper with 2 collections:
  - `course_catalog`: Course metadata for semantic course name matching
  - `course_content`: Chunked lesson content with embeddings
- **`document_processor.py`**: Parses course docs → extracts metadata → sentence-based chunking with overlap
- **`session_manager.py`**: Per-session conversation history (last N exchanges)
- **`config.py`**: Centralized settings loaded from `.env`
- **`models.py`**: Pydantic data models (Course, Lesson, CourseChunk)

### Document Format Expected

Course documents in `docs/` must follow this structure:

```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[lesson content]

Lesson 1: [title]
Lesson Link: [url]
[lesson content]
```

Documents are chunked at **800 characters with 100-character overlap** (sentence boundaries preserved).

## Development Commands

### Setup

```bash
# Install dependencies
uv sync

# Create .env file with required API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application

```bash
# Quick start (from project root)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

Access at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

### Working with the Codebase

```bash
# Run single Python file for testing
cd backend
uv run python script_name.py

# Interactive Python shell with dependencies
uv run python

# Add new dependency
uv add package-name

# Clear ChromaDB (vector database reset)
rm -rf backend/chroma_db
# Documents will be re-indexed on next startup
```

## Key Configuration

Located in `backend/config.py` (loaded from `.env`):

- **`ANTHROPIC_MODEL`**: `claude-sonnet-4-20250514` (default)
- **`EMBEDDING_MODEL`**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **`CHUNK_SIZE`**: 800 characters
- **`CHUNK_OVERLAP`**: 100 characters
- **`MAX_RESULTS`**: 5 (vector search limit)
- **`MAX_HISTORY`**: 2 exchanges (4 messages total in conversation context)
- **`CHROMA_PATH`**: `./chroma_db` (persistent storage)

## Important Implementation Details

### Tool Execution Pattern

The `AIGenerator._handle_tool_execution()` method implements the Anthropic tool use loop:

1. Build messages array: `[user_query, assistant_tool_use, user_tool_results]`
2. Execute all tool calls via `ToolManager.execute_tool()`
3. Make second Claude API call **without tools** to get final response
4. This prevents infinite tool loops

### Vector Search Strategy

`VectorStore.search()` performs **two ChromaDB queries**:

1. **Course resolution** (if `course_name` provided): Semantic search on `course_catalog` collection
2. **Content search**: Query `course_content` with filters (`{"course_title": "...", "lesson_number": ...}`)

This allows partial course name matching (e.g., "MCP" matches "Introduction to MCP Servers").

### Session History Management

`SessionManager` maintains conversation context per session:

- Stores last `MAX_HISTORY * 2` messages (user + assistant pairs)
- History formatted as: `"User: query\nAssistant: response"`
- Included in Claude's system prompt for context-aware responses

### Source Attribution

Sources flow: `ChromaDB metadata` → `CourseSearchTool.last_sources` → `ToolManager.get_last_sources()` → `RAGSystem.query()` → Frontend

Reset after each query to prevent cross-contamination.

## Adding New Course Documents

1. Place `.txt`, `.pdf`, or `.docx` files in `docs/` folder
2. Ensure files follow expected format (see "Document Format Expected" above)
3. Restart application - `app.py` startup event loads documents automatically
4. System skips already-indexed courses (checks `course_catalog` for existing titles)

To force re-indexing: Delete `backend/chroma_db` directory before restart.

## Extending the System

### Adding New Tools

1. Create class inheriting from `search_tools.Tool`
2. Implement `get_tool_definition()` (Anthropic tool schema)
3. Implement `execute(**kwargs)` (tool logic)
4. Register in `RAGSystem.__init__()`: `self.tool_manager.register_tool(YourTool())`

### Modifying Chunking Strategy

Edit `DocumentProcessor.chunk_text()` in `document_processor.py`. Current implementation uses sentence-based chunking with regex: `(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])`

Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py` to change granularity.
- always use uv to run the server, do not use pip directly
- make sure to use uv to manage all dependencies