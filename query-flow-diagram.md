# RAG Chatbot Query Flow Diagram

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAGSystem<br/>(rag_system.py)
    participant Session as SessionManager<br/>(session_manager.py)
    participant AI as AIGenerator<br/>(ai_generator.py)
    participant Claude as Claude API<br/>(Anthropic)
    participant Tools as ToolManager<br/>(search_tools.py)
    participant Search as CourseSearchTool<br/>(search_tools.py)
    participant Vector as VectorStore<br/>(vector_store.py)
    participant DB as ChromaDB<br/>(Vector Database)

    %% Phase 1: User Input
    User->>Frontend: Types "What is prompt caching?"
    User->>Frontend: Clicks Send / Presses Enter

    Note over Frontend: Disable input<br/>Show loading animation
    Frontend->>Frontend: addMessage(query, 'user')
    Frontend->>Frontend: createLoadingMessage()

    %% Phase 2: HTTP Request
    Frontend->>+API: POST /api/query<br/>{query, session_id}

    Note over API: Validate request<br/>QueryRequest model

    %% Phase 3: Session Management
    alt No session_id provided
        API->>+Session: create_session()
        Session-->>-API: "session_1"
    end

    %% Phase 4: RAG Processing
    API->>+RAG: query(query, session_id)

    RAG->>RAG: Build prompt with instructions

    RAG->>+Session: get_conversation_history(session_id)
    Session-->>-RAG: Previous 2 exchanges (4 messages)

    RAG->>+Tools: get_tool_definitions()
    Tools-->>-RAG: [search_course_content definition]

    %% Phase 5: First Claude Call
    RAG->>+AI: generate_response(query, history, tools, tool_manager)

    Note over AI: Build system prompt<br/>+ conversation history

    AI->>+Claude: messages.create()<br/>{prompt, system, tools}

    Note over Claude: Analyzes query<br/>Decides to use tool

    Claude-->>-AI: tool_use response<br/>{name: "search_course_content",<br/>input: {query, course_name}}

    Note over AI: stop_reason == "tool_use"<br/>Handle tool execution

    %% Phase 6: Tool Execution
    AI->>+Tools: execute_tool("search_course_content",<br/>query="prompt caching",<br/>course_name="Building Towards...")

    Tools->>+Search: execute(query, course_name, lesson_number)

    %% Phase 7: Vector Search
    Search->>+Vector: search(query, course_name, lesson_number)

    Note over Vector: Step 1: Resolve course name
    Vector->>+DB: query course_catalog<br/>(semantic match)
    DB-->>-Vector: "Building Towards Computer Use<br/>with Anthropic"

    Note over Vector: Step 2: Build filter<br/>{course_title: "..."}

    Note over Vector: Step 3: Search content
    Vector->>+DB: query course_content<br/>(embedding similarity, n=5)
    DB-->>-Vector: Top 5 chunks with metadata

    Vector-->>-Search: SearchResults(documents, metadata, distances)

    %% Phase 8: Format Results
    Note over Search: Format results with<br/>course & lesson context
    Search->>Search: Store sources internally<br/>(last_sources)

    Search-->>-Tools: "[Course - Lesson X]\ncontent..."
    Tools-->>-AI: Formatted search results

    %% Phase 9: Second Claude Call
    Note over AI: Build messages array:<br/>1. User query<br/>2. Assistant tool_use<br/>3. User tool_result

    AI->>+Claude: messages.create()<br/>{messages with tool results}

    Note over Claude: Synthesizes answer<br/>from search results

    Claude-->>-AI: Final response text
    AI-->>-RAG: "Prompt caching is a feature..."

    %% Phase 10: Extract Sources & Update History
    RAG->>+Tools: get_last_sources()
    Tools-->>-RAG: ["Course - Lesson 3"]

    RAG->>Tools: reset_sources()

    RAG->>+Session: add_exchange(session_id, query, response)
    Note over Session: Store user + assistant messages<br/>Keep last 4 messages (2 exchanges)
    Session-->>-RAG: ✓

    RAG-->>-API: (answer, sources)

    %% Phase 11: API Response
    Note over API: Build QueryResponse<br/>{answer, sources, session_id}

    API-->>-Frontend: JSON response

    %% Phase 12: Display
    Frontend->>Frontend: Remove loading animation
    Frontend->>Frontend: marked.parse(answer)<br/>(Convert markdown to HTML)
    Frontend->>Frontend: addMessage(answer, 'assistant', sources)

    Note over Frontend: Render message with<br/>collapsible sources
    Frontend->>Frontend: Enable input & focus

    Frontend-->>User: Display answer with sources

    Note over User,DB: Total: 2 Claude API calls | 2 ChromaDB queries | ~2-4 seconds
```

## Key Components Interaction Summary

### Three-Phase Architecture

**Phase 1: User Interface Layer**
- Frontend captures input and manages UI state
- Handles loading states and user feedback
- Renders markdown responses with source attribution

**Phase 2: Application Logic Layer**
- FastAPI validates requests and manages sessions
- RAGSystem orchestrates the complete workflow
- SessionManager maintains conversation context

**Phase 3: AI & Data Layer**
- AIGenerator manages Claude API interactions with tool support
- ToolManager coordinates available search capabilities
- VectorStore performs semantic search on ChromaDB
- ChromaDB stores and retrieves vectorized course content

### Critical Interactions

1. **Tool-Based Agent Pattern**: Claude decides whether to search based on query analysis
2. **Two-Step Vector Search**: First resolves course name, then searches content
3. **Session Continuity**: Conversation history flows through entire pipeline
4. **Source Attribution**: Sources tracked from ChromaDB → Search Tool → Frontend

### Performance Characteristics

- **Latency Bottlenecks**:
  - Claude API calls (~1-2s each)
  - Vector embedding & search (~100-500ms)

- **Optimization Strategies**:
  - Temperature=0 for deterministic responses
  - Max 5 search results to limit context
  - Session history capped at 2 exchanges
  - Sentence-based chunking with overlap for better semantic retrieval
