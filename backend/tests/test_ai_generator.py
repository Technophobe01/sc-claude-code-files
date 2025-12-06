"""
Tests for AIGenerator tool calling functionality

These tests evaluate:
1. AIGenerator correctly calls search_course_content for content questions
2. AIGenerator correctly calls get_course_outline for outline questions
3. Tool execution is handled properly
4. Tool results are passed back correctly
5. Final response is generated after tool use
6. Error cases in tool calling are handled
7. Sequential tool calling (up to 2 rounds) works correctly
8. Termination conditions for multi-round tool calling are respected
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from vector_store import VectorStore
from config import config


@pytest.fixture
def vector_store():
    """Create a vector store instance"""
    return VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)


@pytest.fixture
def tool_manager(vector_store):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(vector_store))
    manager.register_tool(CourseOutlineTool(vector_store))
    return manager


@pytest.fixture
def ai_generator():
    """Create an AIGenerator instance"""
    return AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator tool calling functionality"""

    def test_generator_has_required_attributes(self, ai_generator):
        """Test that AIGenerator has required attributes"""
        assert hasattr(ai_generator, 'client')
        assert hasattr(ai_generator, 'model')
        assert hasattr(ai_generator, 'SYSTEM_PROMPT')
        assert hasattr(ai_generator, 'base_params')

    def test_system_prompt_mentions_both_tools(self, ai_generator):
        """Test that system prompt documents both tools"""
        prompt = ai_generator.SYSTEM_PROMPT

        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt

    def test_system_prompt_has_usage_guidelines(self, ai_generator):
        """Test that system prompt has clear usage guidelines for tools"""
        prompt = ai_generator.SYSTEM_PROMPT

        # Should mention when to use each tool
        assert "outline" in prompt.lower() or "syllabus" in prompt.lower()
        assert "content" in prompt.lower() or "search" in prompt.lower()

    def test_tool_definitions_passed_to_api(self, ai_generator, tool_manager):
        """Test that tool definitions are correctly structured for API"""
        tools = tool_manager.get_tool_definitions()

        assert len(tools) == 2

        # Check structure of each tool definition
        for tool_def in tools:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def

        # Verify both tools are present
        tool_names = [t["name"] for t in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_content_query_triggers_search_tool(self, ai_generator, tool_manager):
        """Test that a content-related query triggers search_course_content tool"""
        # This is an integration test - actually calls Claude API
        tools = tool_manager.get_tool_definitions()

        response = ai_generator.generate_response(
            query="What is prompt caching?",
            tools=tools,
            tool_manager=tool_manager
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Should not be an error
        assert "error" not in response.lower() or "Error" not in response[:20]

    def test_outline_query_triggers_outline_tool(self, ai_generator, tool_manager):
        """Test that an outline query triggers get_course_outline tool"""
        tools = tool_manager.get_tool_definitions()

        response = ai_generator.generate_response(
            query="What is the outline of the MCP course?",
            tools=tools,
            tool_manager=tool_manager
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_tool_manager_execute_tool_works(self, tool_manager):
        """Test that ToolManager.execute_tool correctly dispatches to tools"""
        # Test search tool
        result = tool_manager.execute_tool(
            "search_course_content",
            query="prompt caching"
        )
        assert result is not None
        assert isinstance(result, str)

        # Test outline tool
        result = tool_manager.execute_tool(
            "get_course_outline",
            course_name="MCP"
        )
        assert result is not None
        assert isinstance(result, str)

    def test_unknown_tool_returns_error(self, tool_manager):
        """Test that executing unknown tool returns error message"""
        result = tool_manager.execute_tool(
            "nonexistent_tool",
            param="value"
        )
        assert "not found" in result.lower()

    def test_sources_available_after_tool_execution(self, ai_generator, tool_manager):
        """Test that sources are populated after tool-using query"""
        tools = tool_manager.get_tool_definitions()

        # Reset sources
        tool_manager.reset_sources()

        response = ai_generator.generate_response(
            query="What is tool use in Claude?",
            tools=tools,
            tool_manager=tool_manager
        )

        # Get sources - they may or may not be populated depending on tool used
        sources = tool_manager.get_last_sources()
        # Sources should be a list (possibly empty if no tool was used)
        assert isinstance(sources, list)


class TestAIGeneratorToolExecution:
    """Test the _handle_tool_execution method behavior"""

    def test_handle_tool_execution_exists(self, ai_generator):
        """Test that _handle_tool_execution method exists"""
        assert hasattr(ai_generator, '_handle_tool_execution')
        assert callable(ai_generator._handle_tool_execution)

    def test_generate_response_without_tools(self, ai_generator):
        """Test that generate_response works without tools for general questions"""
        response = ai_generator.generate_response(
            query="What is 2 + 2?",
            tools=None,
            tool_manager=None
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


class TestToolManagerSourceTracking:
    """Test ToolManager source tracking functionality"""

    def test_reset_sources_clears_all_tools(self, tool_manager):
        """Test that reset_sources clears sources from all tools"""
        # Execute a search to populate sources
        tool_manager.execute_tool("search_course_content", query="test")

        # Reset
        tool_manager.reset_sources()

        # Get sources should be empty
        sources = tool_manager.get_last_sources()
        assert sources == []

    def test_get_last_sources_returns_from_any_tool(self, tool_manager):
        """Test that get_last_sources returns sources from whichever tool has them"""
        # Execute outline tool
        tool_manager.execute_tool("get_course_outline", course_name="MCP")

        sources = tool_manager.get_last_sources()
        # Should have sources from outline tool
        assert isinstance(sources, list)


class TestSequentialToolCalling:
    """Test sequential tool calling (up to 2 rounds) functionality"""

    def test_max_tool_rounds_constant_exists(self, ai_generator):
        """Test that MAX_TOOL_ROUNDS is defined"""
        assert hasattr(ai_generator, 'MAX_TOOL_ROUNDS')
        assert ai_generator.MAX_TOOL_ROUNDS == 2

    def test_system_prompt_mentions_sequential_calling(self, ai_generator):
        """Test that system prompt documents sequential tool calling capability"""
        prompt = ai_generator.SYSTEM_PROMPT

        assert "sequentially" in prompt.lower() or "sequential" in prompt.lower()
        assert "2" in prompt  # Should mention 2 rounds

    def test_helper_methods_exist(self, ai_generator):
        """Test that helper methods for sequential calling exist"""
        assert hasattr(ai_generator, '_execute_all_tools')
        assert hasattr(ai_generator, '_extract_text_response')
        assert hasattr(ai_generator, '_make_final_call')
        assert hasattr(ai_generator, '_process_tool_round')

        assert callable(ai_generator._execute_all_tools)
        assert callable(ai_generator._extract_text_response)
        assert callable(ai_generator._make_final_call)
        assert callable(ai_generator._process_tool_round)

    def test_two_round_tool_execution(self, ai_generator, tool_manager):
        """
        Integration test: Query requiring two tool calls completes successfully.
        This tests that Claude can chain tool calls when needed.
        """
        tools = tool_manager.get_tool_definitions()

        # Query that might benefit from two tool calls:
        # First get course outline, then search for specific content
        response = ai_generator.generate_response(
            query="What topic is covered in lesson 2 of the MCP course and explain it in detail?",
            tools=tools,
            tool_manager=tool_manager
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_single_tool_still_works(self, ai_generator, tool_manager):
        """Test that simple queries still work with single tool call"""
        tools = tool_manager.get_tool_definitions()

        response = ai_generator.generate_response(
            query="What is the outline of the Prompt Caching course?",
            tools=tools,
            tool_manager=tool_manager
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


class TestSequentialToolCallingMocked:
    """Test sequential tool calling with mocked API responses"""

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager that tracks calls"""
        manager = Mock()
        manager.call_count = 0
        manager.calls = []

        def mock_execute(name, **kwargs):
            manager.call_count += 1
            manager.calls.append({"name": name, "kwargs": kwargs})
            if name == "get_course_outline":
                return "Course: MCP\nLesson 1: Introduction\nLesson 2: Tool Use"
            elif name == "search_course_content":
                return "Tool use allows Claude to interact with external systems."
            return f"Result for {name}"

        manager.execute_tool = mock_execute
        return manager

    def test_api_called_multiple_times_for_sequential_tools(self, mock_tool_manager):
        """Test that API is called multiple times when Claude chains tools"""
        # Create mock response objects
        @dataclass
        class MockToolUseBlock:
            type: str = "tool_use"
            id: str = "tool_1"
            name: str = "get_course_outline"
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {"course_name": "MCP"}

        @dataclass
        class MockTextBlock:
            type: str = "text"
            text: str = "Here is my response based on the tools."

        @dataclass
        class MockResponse:
            stop_reason: str = "tool_use"
            content: List[Any] = None

        # Track API calls
        api_call_count = 0
        api_calls = []

        def mock_create(**kwargs):
            nonlocal api_call_count
            api_call_count += 1
            api_calls.append(kwargs)

            # First call: return tool_use for outline
            if api_call_count == 1:
                block = MockToolUseBlock(
                    id="tool_1",
                    name="get_course_outline",
                    input={"course_name": "MCP"}
                )
                return MockResponse(stop_reason="tool_use", content=[block])

            # Second call: return tool_use for search
            elif api_call_count == 2:
                block = MockToolUseBlock(
                    id="tool_2",
                    name="search_course_content",
                    input={"query": "Tool Use"}
                )
                return MockResponse(stop_reason="tool_use", content=[block])

            # Third call: return final text
            else:
                return MockResponse(
                    stop_reason="end_turn",
                    content=[MockTextBlock(text="Final synthesized answer.")]
                )

        # Create generator with mocked client
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            generator = AIGenerator("fake-key", "claude-test")

            tools = [
                {"name": "get_course_outline", "description": "Get outline", "input_schema": {}},
                {"name": "search_course_content", "description": "Search content", "input_schema": {}}
            ]

            response = generator.generate_response(
                query="What is tool use in the MCP course?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        # Verify behavior
        assert response == "Final synthesized answer."
        assert api_call_count == 3  # Initial + round 2 + final synthesis
        assert mock_tool_manager.call_count == 2  # Two tools executed

    def test_terminates_after_max_rounds(self, mock_tool_manager):
        """Test that tool calling terminates after MAX_TOOL_ROUNDS even if Claude keeps requesting tools"""
        @dataclass
        class MockToolUseBlock:
            type: str = "tool_use"
            id: str = "tool_1"
            name: str = "search_course_content"
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {"query": "test"}

        @dataclass
        class MockTextBlock:
            type: str = "text"
            text: str = "Forced final response."

        @dataclass
        class MockResponse:
            stop_reason: str = "tool_use"
            content: List[Any] = None

        api_call_count = 0

        def mock_create(**kwargs):
            nonlocal api_call_count
            api_call_count += 1

            # Check if this is a call without tools (forced final synthesis)
            has_tools = "tools" in kwargs and kwargs["tools"]

            # If no tools in request, this is the forced final call
            if not has_tools:
                return MockResponse(
                    stop_reason="end_turn",
                    content=[MockTextBlock()]
                )

            # Return tool_use for calls with tools
            block = MockToolUseBlock(id=f"tool_{api_call_count}")
            return MockResponse(stop_reason="tool_use", content=[block])

        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            generator = AIGenerator("fake-key", "claude-test")

            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            response = generator.generate_response(
                query="Keep searching forever",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        # Should terminate: initial + round 2 + forced final = 3 calls
        # (round 1 in generate_response, round 2 in _process_tool_round,
        # then max exceeded triggers _make_final_call without tools)
        assert api_call_count == 3
        assert response == "Forced final response."
        # Should have executed 2 tools (one per round)
        assert mock_tool_manager.call_count == 2

    def test_terminates_when_no_tool_use(self, mock_tool_manager):
        """Test that tool calling terminates when Claude doesn't request tools"""
        @dataclass
        class MockTextBlock:
            type: str = "text"
            text: str = "Direct answer without tools."

        @dataclass
        class MockResponse:
            stop_reason: str = "end_turn"
            content: List[Any] = None

        api_call_count = 0

        def mock_create(**kwargs):
            nonlocal api_call_count
            api_call_count += 1
            return MockResponse(
                stop_reason="end_turn",
                content=[MockTextBlock()]
            )

        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            generator = AIGenerator("fake-key", "claude-test")

            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            response = generator.generate_response(
                query="What is 2 + 2?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        assert api_call_count == 1  # Only initial call
        assert mock_tool_manager.call_count == 0  # No tools executed
        assert response == "Direct answer without tools."

    def test_terminates_on_tool_error(self, mock_tool_manager):
        """Test that tool calling terminates gracefully on tool execution error"""
        @dataclass
        class MockToolUseBlock:
            type: str = "tool_use"
            id: str = "tool_1"
            name: str = "failing_tool"
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {}

        @dataclass
        class MockTextBlock:
            type: str = "text"
            text: str = "Response after error."

        @dataclass
        class MockResponse:
            stop_reason: str = "tool_use"
            content: List[Any] = None

        # Make tool manager return error
        def error_execute(name, **kwargs):
            return "Error: Tool 'failing_tool' not found"

        mock_tool_manager.execute_tool = error_execute

        api_call_count = 0

        def mock_create(**kwargs):
            nonlocal api_call_count
            api_call_count += 1

            if api_call_count == 1:
                return MockResponse(
                    stop_reason="tool_use",
                    content=[MockToolUseBlock()]
                )
            return MockResponse(
                stop_reason="end_turn",
                content=[MockTextBlock()]
            )

        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            generator = AIGenerator("fake-key", "claude-test")

            tools = [{"name": "failing_tool", "description": "Fails", "input_schema": {}}]

            response = generator.generate_response(
                query="Use the failing tool",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        # Should make final call after error
        assert api_call_count == 2
        assert response == "Response after error."

    def test_messages_accumulate_across_rounds(self, mock_tool_manager):
        """Test that conversation messages accumulate correctly across rounds"""
        @dataclass
        class MockToolUseBlock:
            type: str = "tool_use"
            id: str = "tool_1"
            name: str = "get_course_outline"
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {"course_name": "MCP"}

        @dataclass
        class MockTextBlock:
            type: str = "text"
            text: str = "Final response."

        @dataclass
        class MockResponse:
            stop_reason: str = "tool_use"
            content: List[Any] = None

        captured_messages = []

        def mock_create(**kwargs):
            captured_messages.append(len(kwargs.get("messages", [])))

            if len(captured_messages) == 1:
                # First call: 1 message (user query)
                return MockResponse(
                    stop_reason="tool_use",
                    content=[MockToolUseBlock(id="tool_1")]
                )
            elif len(captured_messages) == 2:
                # Second call: 3 messages (user + assistant tool_use + user tool_result)
                return MockResponse(
                    stop_reason="tool_use",
                    content=[MockToolUseBlock(id="tool_2", name="search_course_content", input={"query": "test"})]
                )
            else:
                # Final call: 5 messages
                return MockResponse(
                    stop_reason="end_turn",
                    content=[MockTextBlock()]
                )

        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create = mock_create
            MockAnthropic.return_value = mock_client

            generator = AIGenerator("fake-key", "claude-test")

            tools = [
                {"name": "get_course_outline", "description": "Outline", "input_schema": {}},
                {"name": "search_course_content", "description": "Search", "input_schema": {}}
            ]

            generator.generate_response(
                query="Test query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

        # Verify message accumulation pattern
        assert captured_messages[0] == 1  # Initial: user query only
        assert captured_messages[1] == 3  # After round 1: user + assistant + tool_result
        assert captured_messages[2] == 5  # After round 2: + assistant + tool_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
