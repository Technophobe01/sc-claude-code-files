import anthropic
from typing import List, Optional, Dict, Any, Tuple

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    # Maximum number of sequential tool call rounds per query
    MAX_TOOL_ROUNDS = 2

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search for specific content within course materials
   - Use for questions about specific topics, concepts, or lesson details
2. **get_course_outline**: Get complete course structure with all lessons
   - Use for questions about course outlines, syllabi, what topics are covered, or lesson lists
   - Returns: course title, course link, and complete lesson list (lesson number and title for each)

Tool Usage Guidelines:
- Use **get_course_outline** for outline/syllabus/structure questions (e.g., "What is the outline of...", "What topics are covered in...", "List the lessons in...")
- Use **search_course_content** for specific content questions (e.g., "What is prompt caching?", "How does tool use work?")
- You may call tools sequentially (up to 2 rounds) when a query requires multiple steps
- Use multi-step tool calls when:
  - You need course structure first to narrow a subsequent content search
  - A query involves comparing information from different sources
  - The answer requires combining results from multiple tool calls
- Prefer single tool calls when sufficient for the query
- Synthesize all tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline, then present the course title, link, and all lessons
- **Course content questions**: Use search_course_content, then answer
- **Multi-step questions**: Chain tool calls as needed (e.g., get outline first, then search specific content)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the tool results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def _execute_all_tools(self, response, tool_manager) -> Tuple[List[Dict], bool]:
        """
        Execute all tool_use blocks in a response.

        Args:
            response: Claude API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (list of tool_result dicts, has_error boolean)
        """
        tool_results = []
        has_error = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    # Check for error indicators in result
                    if isinstance(result, str) and ("not found" in result.lower() or
                                                     result.startswith("Error")):
                        has_error = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })
                except Exception as e:
                    has_error = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })

        return tool_results, has_error

    def _extract_text_response(self, response) -> str:
        """
        Extract text content from a Claude API response.

        Args:
            response: Claude API response

        Returns:
            Text content from the response, or fallback message
        """
        if response.content:
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
        return "I couldn't generate a response based on the available information."

    def _make_final_call(self, messages: List[Dict], system_prompt: str) -> str:
        """
        Make a final API call without tools to force text synthesis.

        Args:
            messages: Accumulated conversation messages
            system_prompt: System prompt to use

        Returns:
            Final text response
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_prompt
        }
        response = self.client.messages.create(**final_params)
        return self._extract_text_response(response)

    def _process_tool_round(self,
                            messages: List[Dict],
                            system_prompt: str,
                            tools: List,
                            tool_manager,
                            round_number: int) -> str:
        """
        Recursively process tool rounds until termination condition.

        Termination conditions:
        - round_number > MAX_TOOL_ROUNDS
        - response.stop_reason != "tool_use"
        - tool execution failure

        Args:
            messages: Accumulated conversation messages
            system_prompt: System prompt to use
            tools: Tool definitions for API call
            tool_manager: Manager to execute tools
            round_number: Current round (1-indexed)

        Returns:
            Final text response after all tool rounds complete
        """
        # BASE CASE 1: Max rounds exceeded - force final synthesis
        if round_number > self.MAX_TOOL_ROUNDS:
            return self._make_final_call(messages, system_prompt)

        # Make API call WITH tools (enables continued tool use)
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_prompt,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }

        response = self.client.messages.create(**api_params)

        # BASE CASE 2: No tool use - return text response
        if response.stop_reason != "tool_use":
            return self._extract_text_response(response)

        # Execute tool calls
        tool_results, has_error = self._execute_all_tools(response, tool_manager)

        # BASE CASE 3: Tool execution failed - make final call with partial results
        if has_error:
            # Still add the results so Claude can see what happened
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            return self._make_final_call(messages, system_prompt)

        # RECURSIVE CASE: Build updated messages and continue
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        return self._process_tool_round(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_manager=tool_manager,
            round_number=round_number + 1
        )

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(
                response, api_params, tool_manager, tools=tools
            )

        # Return direct response
        return self._extract_text_response(response)

    def _handle_tool_execution(self,
                               initial_response,
                               base_params: Dict[str, Any],
                               tool_manager,
                               tools: Optional[List] = None):
        """
        Entry point for tool execution. Delegates to recursive helper for
        sequential tool calling support.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters including messages and system prompt
            tool_manager: Manager to execute tools
            tools: Tool definitions for continued tool use in subsequent rounds

        Returns:
            Final response text after all tool rounds complete
        """
        # Build initial messages with user query + assistant tool_use
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute tools from initial response
        tool_results, has_error = self._execute_all_tools(initial_response, tool_manager)

        # Add tool results to messages
        messages.append({"role": "user", "content": tool_results})

        # If no tools provided or error occurred, make final call without tools
        if not tools or has_error:
            return self._make_final_call(messages, base_params["system"])

        # Delegate to recursive processor starting at round 2
        # (round 1 was the initial response that triggered this method)
        return self._process_tool_round(
            messages=messages,
            system_prompt=base_params["system"],
            tools=tools,
            tool_manager=tool_manager,
            round_number=2
        )