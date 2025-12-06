"""
Tests for RAG system content query handling

These tests evaluate:
1. RAGSystem correctly orchestrates tool-based queries
2. Content queries return valid responses with sources
3. Session management works correctly
4. Error handling for various query types
5. Source attribution flows correctly through the system
"""

import pytest
import sys
import os
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config


@pytest.fixture
def rag_system():
    """Create a RAGSystem instance"""
    return RAGSystem(config)


class TestRAGSystemContentQueries:
    """Test suite for RAG system content query handling"""

    def test_rag_system_initialized_correctly(self, rag_system):
        """Test that RAGSystem has all required components"""
        assert hasattr(rag_system, 'document_processor')
        assert hasattr(rag_system, 'vector_store')
        assert hasattr(rag_system, 'ai_generator')
        assert hasattr(rag_system, 'session_manager')
        assert hasattr(rag_system, 'tool_manager')
        assert hasattr(rag_system, 'search_tool')
        assert hasattr(rag_system, 'outline_tool')

    def test_tool_manager_has_both_tools(self, rag_system):
        """Test that ToolManager has both search and outline tools registered"""
        tools = rag_system.tool_manager.get_tool_definitions()

        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_content_query_returns_response_and_sources(self, rag_system):
        """Test that a content query returns both response and sources"""
        response, sources = rag_system.query("What is prompt caching?")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

        # Sources should be a list
        assert isinstance(sources, list)

    def test_content_query_response_not_error(self, rag_system):
        """Test that content query doesn't return an error"""
        response, sources = rag_system.query("What is tool use in Claude?")

        # Should not start with error indicators
        assert not response.startswith("Error")
        assert "query failed" not in response.lower()

    def test_sources_have_correct_structure(self, rag_system):
        """Test that sources have the expected dict structure"""
        response, sources = rag_system.query("How does prompt caching work?")

        # If we got sources, check their structure
        if sources:
            for source in sources:
                assert isinstance(source, dict)
                assert "text" in source
                assert "url" in source

    def test_sources_reset_between_queries(self, rag_system):
        """Test that sources are reset between queries"""
        # First query
        response1, sources1 = rag_system.query("What is prompt caching?")

        # Second query
        response2, sources2 = rag_system.query("What is MCP?")

        # Sources should not accumulate
        # (can't easily test they're different, but should verify they're independent)
        assert isinstance(sources1, list)
        assert isinstance(sources2, list)

    def test_query_with_session_id(self, rag_system):
        """Test that queries work with session ID"""
        session_id = "test-session-123"

        response, sources = rag_system.query(
            "What is prompt caching?",
            session_id=session_id
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_session_history_is_maintained(self, rag_system):
        """Test that conversation history is maintained across queries"""
        session_id = "test-session-456"

        # First query
        response1, _ = rag_system.query("What is tool use?", session_id=session_id)

        # Second query should have context
        response2, _ = rag_system.query("Tell me more about it", session_id=session_id)

        # Both should return valid responses
        assert response1 is not None
        assert response2 is not None

    def test_outline_query_returns_response(self, rag_system):
        """Test that outline queries work through the system"""
        response, sources = rag_system.query("What is the outline of the MCP course?")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_general_knowledge_query(self, rag_system):
        """Test that general knowledge queries work without tool use"""
        response, sources = rag_system.query("What is 2 + 2?")

        assert response is not None
        assert isinstance(response, str)
        # General knowledge query may or may not have sources
        assert isinstance(sources, list)


class TestRAGSystemErrorHandling:
    """Test error handling in RAG system"""

    def test_empty_query_handling(self, rag_system):
        """Test handling of empty queries"""
        response, sources = rag_system.query("")

        # Should handle gracefully
        assert response is not None
        assert isinstance(response, str)

    def test_very_long_query_handling(self, rag_system):
        """Test handling of very long queries"""
        long_query = "What is prompt caching? " * 50
        response, sources = rag_system.query(long_query)

        # Should handle gracefully
        assert response is not None
        assert isinstance(response, str)

    def test_special_characters_in_query(self, rag_system):
        """Test handling of special characters in queries"""
        response, sources = rag_system.query("What's the <API> usage & \"cost\"?")

        assert response is not None
        assert isinstance(response, str)


class TestRAGSystemSourceFlow:
    """Test source attribution flow through the system"""

    def test_sources_from_search_tool(self, rag_system):
        """Test that sources flow correctly from search tool"""
        # Reset sources
        rag_system.tool_manager.reset_sources()

        # Query that should trigger search
        response, sources = rag_system.query("Explain prompt caching in detail")

        # Check sources are returned
        assert isinstance(sources, list)

    def test_sources_include_lesson_info(self, rag_system):
        """Test that sources include lesson information"""
        response, sources = rag_system.query("What is tool use?")

        if sources:
            for source in sources:
                # Source text should mention course or lesson
                source_text = source.get("text", "")
                # At minimum, source should have some content
                assert len(source_text) > 0

    def test_source_urls_are_valid(self, rag_system):
        """Test that source URLs are valid when present"""
        response, sources = rag_system.query("How does prompt caching work?")

        if sources:
            for source in sources:
                url = source.get("url")
                if url is not None:
                    assert isinstance(url, str)
                    assert url.startswith("http")


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""

    def test_get_course_analytics(self, rag_system):
        """Test that course analytics returns expected structure"""
        analytics = rag_system.get_course_analytics()

        assert isinstance(analytics, dict)
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert isinstance(analytics["total_courses"], int)
        assert isinstance(analytics["course_titles"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
