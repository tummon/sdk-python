"""Lazy tool description loading - reduces context by deferring full tool descriptions.

This module provides a handler that abbreviates tool descriptions sent to the model,
allowing tools to be expanded on-demand via the expand_tool meta-tool. This can
significantly reduce context window usage when working with many tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .decorator import tool
from ..types.tools import ToolSpec

if TYPE_CHECKING:
    pass


@dataclass
class LazyToolHandler:
    """Handles lazy loading of tool descriptions to save context.

    When enabled, tools are sent to the model with abbreviated descriptions.
    The model can call `expand_tool(name)` to get full details before use.

    Usage:
        handler = LazyToolHandler()
        agent = Agent(tools=[...], lazy_tool_handler=handler)

    Attributes:
        require_expansion: If True, block execution of unexpanded tools (safer but stricter).
        summary_max_length: Maximum characters for auto-generated summaries.
    """

    require_expansion: bool = True
    summary_max_length: int = 100

    # Track full specs for expansion (internal)
    _full_specs: dict[str, ToolSpec] = field(default_factory=dict)

    def transform(
        self,
        tool_specs: list[ToolSpec],
        invocation_state: dict[str, Any],
    ) -> list[ToolSpec]:
        """Transform full tool specs into abbreviated versions.

        Called at event_loop.py - the single injection point.
        Works for ALL tools including MCP tools.

        Args:
            tool_specs: Full tool specifications from the registry.
            invocation_state: Current invocation state dict.

        Returns:
            List of abbreviated tool specs plus the expand_tool spec.
        """
        # Cache full specs for later expansion
        self._full_specs = {spec["name"]: spec for spec in tool_specs}

        # Get already-expanded tools from state
        expanded: set[str] = invocation_state.get("_expanded_tools", set())

        result = []
        for spec in tool_specs:
            name = spec["name"]
            if name in expanded:
                # Already expanded - send full spec
                result.append(spec)
            else:
                # Send abbreviated spec
                result.append(self._abbreviate(spec))

        # Add our expand_tool (always with full spec)
        result.append(self._get_expand_tool_spec())

        return result

    def _abbreviate(self, spec: ToolSpec) -> ToolSpec:
        """Create abbreviated spec with just name and summary."""
        summary = self._extract_summary(spec.get("description", ""))
        return {
            "name": spec["name"],
            "description": f"[Abbreviated] {summary} Use expand_tool('{spec['name']}') for full details.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }

    def _extract_summary(self, description: str) -> str:
        """Extract first sentence as summary."""
        if not description:
            return "No description available."

        # Get first sentence
        first_sentence = description.split(". ")[0].strip()

        # Truncate if needed
        if len(first_sentence) > self.summary_max_length:
            return first_sentence[: self.summary_max_length - 3] + "..."

        return first_sentence + ("." if not first_sentence.endswith(".") else "")

    def expand(self, tool_name: str, invocation_state: dict[str, Any]) -> str:
        """Expand a tool to get its full description.

        Called by the expand_tool tool.

        Args:
            tool_name: Name of the tool to expand.
            invocation_state: Current invocation state dict.

        Returns:
            Formatted full tool specification as a string.
        """
        if tool_name not in self._full_specs:
            available = ", ".join(sorted(self._full_specs.keys()))
            return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

        # Mark as expanded
        expanded: set[str] = invocation_state.setdefault("_expanded_tools", set())
        expanded.add(tool_name)

        # Return formatted full spec
        spec = self._full_specs[tool_name]
        return self._format_spec(spec)

    def _format_spec(self, spec: ToolSpec) -> str:
        """Format a tool spec for the model to read."""
        schema_str = json.dumps(spec.get("inputSchema", {}), indent=2)
        return f"""## Tool: {spec['name']}

### Description
{spec.get('description', 'No description')}

### Parameters
```json
{schema_str}
```

You can now call this tool with the parameters shown above."""

    def _get_expand_tool_spec(self) -> ToolSpec:
        """Get the spec for expand_tool itself."""
        return {
            "name": "expand_tool",
            "description": (
                "Get the full description and parameter schema for a tool. "
                "Call this BEFORE using any tool to see its complete documentation and required parameters."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "The name of the tool to expand"}
                    },
                    "required": ["tool_name"],
                }
            },
        }

    def should_block_unexpanded(self, tool_name: str, invocation_state: dict[str, Any]) -> bool:
        """Check if a tool call should be blocked because it wasn't expanded.

        Args:
            tool_name: Name of the tool being called.
            invocation_state: Current invocation state dict.

        Returns:
            True if the tool call should be blocked, False otherwise.
        """
        if not self.require_expansion:
            return False
        if tool_name == "expand_tool":
            return False
        expanded: set[str] = invocation_state.get("_expanded_tools", set())
        return tool_name not in expanded

    def get_block_message(self, tool_name: str) -> str:
        """Get the message to return when blocking an unexpanded tool.

        Args:
            tool_name: Name of the tool that was blocked.

        Returns:
            Error message instructing the model to expand the tool first.
        """
        return (
            f"This tool has not been expanded yet. Please call expand_tool('{tool_name}') "
            f"first to see the full description and required parameters, then try again."
        )


def create_expand_tool(handler: LazyToolHandler):
    """Create the expand_tool function bound to a handler.

    Args:
        handler: The LazyToolHandler instance to bind to.

    Returns:
        A decorated tool function that expands tool descriptions.
    """

    @tool(name="expand_tool")
    def expand_tool(
        tool_name: str,
        _invocation_state: dict[str, Any],
    ) -> str:
        """Get the full description and parameter schema for a tool.

        Args:
            tool_name: The name of the tool to expand

        Returns:
            Complete tool documentation including description and parameters
        """
        return handler.expand(tool_name, _invocation_state)

    return expand_tool
