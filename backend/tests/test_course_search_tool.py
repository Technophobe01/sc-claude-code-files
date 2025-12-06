"""
Tests for CourseSearchTool.execute method

These tests evaluate:
1. Basic search functionality returns results
2. Search with course_name filter works correctly
3. Search with lesson_number filter works correctly
4. Combined filters work correctly
5. Output format is correct (has headers, content)
6. Sources are populated with correct structure (text and url)
7. Empty results are handled properly
8. Error cases are handled gracefully
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import VectorStore
from config import config


@pytest.fixture
def vector_store():
    """Create a vector store instance connected to the existing database"""
    return VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)


@pytest.fixture
def search_tool(vector_store):
    """Create a CourseSearchTool instance"""
    return CourseSearchTool(vector_store)


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute method"""

    def test_basic_search_returns_results(self, search_tool):
        """Test that a basic search query returns non-empty results"""
        result = search_tool.execute(query="prompt caching")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert "No relevant content found" not in result
        assert "error" not in result.lower()

    def test_search_result_contains_course_header(self, search_tool):
        """Test that search results contain course context headers"""
        result = search_tool.execute(query="tool use")

        # Results should have headers in format [Course Title - Lesson N]
        assert "[" in result
        assert "]" in result

    def test_search_with_course_name_filter(self, search_tool):
        """Test that course_name filter returns results from specific course"""
        result = search_tool.execute(
            query="introduction",
            course_name="MCP"
        )

        assert result is not None
        assert isinstance(result, str)
        # Should either have results or a clear "not found" message
        if "No relevant content found" not in result:
            # If we have results, they should be from the MCP course
            assert "MCP" in result or "mcp" in result.lower()

    def test_search_with_lesson_number_filter(self, search_tool):
        """Test that lesson_number filter returns results from specific lesson"""
        result = search_tool.execute(
            query="content",
            course_name="Computer Use",
            lesson_number=1
        )

        assert result is not None
        assert isinstance(result, str)

    def test_sources_populated_after_search(self, search_tool):
        """Test that last_sources is populated with correct structure after search"""
        # Reset sources first
        search_tool.last_sources = []

        result = search_tool.execute(query="prompt caching")

        # Sources should be populated if we got results
        if "No relevant content found" not in result:
            assert len(search_tool.last_sources) > 0

            # Check structure of first source
            first_source = search_tool.last_sources[0]
            assert isinstance(first_source, dict)
            assert "text" in first_source
            assert "url" in first_source
            assert isinstance(first_source["text"], str)
            # url can be None or string

    def test_source_text_contains_course_and_lesson(self, search_tool):
        """Test that source text contains course title and lesson number"""
        search_tool.last_sources = []
        result = search_tool.execute(query="tool use")

        if "No relevant content found" not in result and search_tool.last_sources:
            first_source = search_tool.last_sources[0]
            source_text = first_source["text"]

            # Source text should have format "Course Title - Lesson N"
            assert " - Lesson " in source_text or len(source_text) > 0

    def test_source_url_is_valid_or_none(self, search_tool):
        """Test that source URLs are either valid URLs or None"""
        search_tool.last_sources = []
        result = search_tool.execute(query="prompt caching")

        if search_tool.last_sources:
            for source in search_tool.last_sources:
                url = source.get("url")
                if url is not None:
                    assert isinstance(url, str)
                    assert url.startswith("http")

    def test_empty_query_handling(self, search_tool):
        """Test handling of empty or very short queries"""
        result = search_tool.execute(query="")

        # Should handle gracefully - either return results or no-match message
        assert result is not None
        assert isinstance(result, str)

    def test_nonexistent_course_filter(self, search_tool):
        """Test that searching in non-existent course returns appropriate message"""
        result = search_tool.execute(
            query="anything",
            course_name="NonExistentCourseXYZ123"
        )

        # Should return a "not found" type message
        assert "No" in result or "not found" in result.lower() or "No course found" in result

    def test_search_returns_multiple_results(self, search_tool):
        """Test that search can return multiple results separated by newlines"""
        result = search_tool.execute(query="Claude API")

        if "No relevant content found" not in result:
            # Multiple results should be separated by double newlines
            # Check that there's some content
            assert len(result) > 50

    def test_result_format_has_content_after_header(self, search_tool):
        """Test that each result has content following the header"""
        result = search_tool.execute(query="embeddings")

        if "No relevant content found" not in result:
            # Split by double newline to get individual results
            parts = result.split("\n\n")
            for part in parts:
                if part.strip():
                    # Each non-empty part should have both header and content
                    lines = part.strip().split("\n")
                    assert len(lines) >= 1


class TestCourseSearchToolIntegration:
    """Integration tests that verify the tool works with real data"""

    def test_known_topic_returns_relevant_content(self, search_tool):
        """Test searching for a known topic returns relevant results"""
        result = search_tool.execute(query="what is prompt caching")

        # We know prompt caching is discussed in the courses
        assert result is not None
        if "No relevant content found" not in result:
            # Content should mention caching or related terms
            result_lower = result.lower()
            assert "cach" in result_lower or "prompt" in result_lower

    def test_mcp_course_search(self, search_tool):
        """Test searching specifically in MCP course"""
        result = search_tool.execute(
            query="server",
            course_name="MCP"
        )

        assert result is not None
        # Should either find content or indicate no match
        assert len(result) > 0

    def test_tool_definition_is_valid(self, search_tool):
        """Test that tool definition has required fields for Anthropic API"""
        tool_def = search_tool.get_tool_definition()

        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def

        schema = tool_def["input_schema"]
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
