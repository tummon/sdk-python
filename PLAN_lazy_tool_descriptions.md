# Implementation Plan: Lazy Tool Description Loading (Simplified)

## Problem Statement

When an agent has many tools, sending full descriptions for all tools in every API call consumes significant context window space and increases token costs.

## Key Discovery

**There is ONE single injection point** at `event_loop.py:339` where ALL tool specs (regular + MCP) pass through before going to the model. This enables a fully encapsulated solution.

---

## Simplified Design: Standalone Handler Class

Create a single `LazyToolHandler` class that:
1. Lives in ONE new file
2. Contains ALL logic (transformation, expand_tool, state tracking)
3. Is optionally passed to Agent
4. When `None`, zero code paths are affected

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  event_loop.py:339                                          │
│                                                             │
│  tool_specs = agent.tool_registry.get_all_tool_specs()      │
│                                                             │
│  # NEW: Single injection point                              │
│  if agent.lazy_tool_handler:                                │
│      tool_specs = agent.lazy_tool_handler.transform(        │
│          tool_specs, invocation_state                       │
│      )                                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LazyToolHandler (single file, fully self-contained)        │
│                                                             │
│  - transform(specs, state) → abbreviated specs + expand_tool│
│  - expand_tool implementation (registered as tool)          │
│  - summary extraction logic                                 │
│  - expanded state tracking                                  │
│  - optional: block unexpanded tool calls                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation

### File 1: `src/strands/tools/lazy_tool_handler.py` (NEW - all logic here)

```python
"""Lazy tool description loading - reduces context by deferring full tool descriptions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from strands.tools.decorator import tool
from strands.types.tools import ToolSpec

if TYPE_CHECKING:
    from strands.agent import Agent


@dataclass
class LazyToolHandler:
    """
    Handles lazy loading of tool descriptions to save context.

    When enabled, tools are sent to the model with abbreviated descriptions.
    The model can call `expand_tool(name)` to get full details before use.

    Usage:
        handler = LazyToolHandler()
        agent = Agent(tools=[...], lazy_tool_handler=handler)
    """

    # If True, block execution of unexpanded tools (safer but stricter)
    require_expansion: bool = True

    # Max chars for auto-generated summaries
    summary_max_length: int = 100

    # Track full specs for expansion
    _full_specs: dict[str, ToolSpec] = field(default_factory=dict)

    def transform(
        self,
        tool_specs: list[ToolSpec],
        invocation_state: dict[str, Any]
    ) -> list[ToolSpec]:
        """
        Transform full tool specs into abbreviated versions.

        Called at event_loop.py:339 - the single injection point.
        Works for ALL tools including MCP tools.
        """
        # Cache full specs for later expansion
        self._full_specs = {spec["name"]: spec for spec in tool_specs}

        # Get already-expanded tools from state
        expanded = invocation_state.get("_expanded_tools", set())

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
            return first_sentence[:self.summary_max_length - 3] + "..."

        return first_sentence + ("." if not first_sentence.endswith(".") else "")

    def expand(self, tool_name: str, invocation_state: dict[str, Any]) -> str:
        """
        Expand a tool to get its full description.
        Called by the expand_tool tool.
        """
        if tool_name not in self._full_specs:
            available = ", ".join(sorted(self._full_specs.keys()))
            return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

        # Mark as expanded
        expanded = invocation_state.setdefault("_expanded_tools", set())
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
                        "tool_name": {
                            "type": "string",
                            "description": "The name of the tool to expand"
                        }
                    },
                    "required": ["tool_name"]
                }
            }
        }

    def should_block_unexpanded(self, tool_name: str, invocation_state: dict[str, Any]) -> bool:
        """Check if a tool call should be blocked because it wasn't expanded."""
        if not self.require_expansion:
            return False
        if tool_name == "expand_tool":
            return False
        expanded = invocation_state.get("_expanded_tools", set())
        return tool_name not in expanded

    def get_block_message(self, tool_name: str) -> str:
        """Get the message to return when blocking an unexpanded tool."""
        return (
            f"This tool has not been expanded yet. Please call expand_tool('{tool_name}') "
            f"first to see the full description and required parameters, then try again."
        )


# The actual tool function that gets registered
def create_expand_tool(handler: LazyToolHandler):
    """Create the expand_tool function bound to a handler."""

    @tool(name="expand_tool")
    def expand_tool(
        tool_name: str,
        _invocation_state: dict[str, Any],
    ) -> str:
        """
        Get the full description and parameter schema for a tool.

        Args:
            tool_name: The name of the tool to expand

        Returns:
            Complete tool documentation including description and parameters
        """
        return handler.expand(tool_name, _invocation_state)

    return expand_tool
```

---

### File 2: `src/strands/agent/agent.py` (minimal change)

```python
# In __init__, add ONE parameter:
def __init__(
    self,
    ...,
    lazy_tool_handler: "LazyToolHandler | None" = None,  # NEW
):
    ...
    self.lazy_tool_handler = lazy_tool_handler

    # Register expand_tool if handler provided
    if lazy_tool_handler:
        from strands.tools.lazy_tool_handler import create_expand_tool
        self.tool_registry.register_tool(create_expand_tool(lazy_tool_handler))
```

---

### File 3: `src/strands/event_loop/event_loop.py` (single injection point)

```python
# Around line 339, change:
tool_specs = agent.tool_registry.get_all_tool_specs()

# To:
tool_specs = agent.tool_registry.get_all_tool_specs()
if agent.lazy_tool_handler:
    tool_specs = agent.lazy_tool_handler.transform(tool_specs, invocation_state)
```

---

### File 4: `src/strands/tools/executors/_executor.py` (optional blocking)

```python
# At the start of _stream(), add:
if agent.lazy_tool_handler and agent.lazy_tool_handler.should_block_unexpanded(tool_name, invocation_state):
    yield ToolResultEvent(
        tool_use_id=tool_use["toolUseId"],
        status="error",
        content=agent.lazy_tool_handler.get_block_message(tool_name)
    )
    return
```

---

## Usage

```python
from strands import Agent
from strands.tools.lazy_tool_handler import LazyToolHandler

# Enable lazy loading
handler = LazyToolHandler(require_expansion=True)
agent = Agent(
    tools=[tool1, tool2, ...many_tools...],
    lazy_tool_handler=handler
)

# Disabled (default) - no code paths affected
agent = Agent(tools=[tool1, tool2, ...])  # lazy_tool_handler=None
```

---

## Why This Design is Clean

| Aspect | Benefit |
|--------|---------|
| **Single new file** | All logic in `lazy_tool_handler.py` |
| **Single injection point** | Only `event_loop.py:339` transforms specs |
| **Works for ALL tools** | MCP tools pass through same point |
| **Zero impact when disabled** | `if agent.lazy_tool_handler:` guards everything |
| **No type changes** | Uses existing `ToolSpec` type |
| **No decorator changes** | Auto-extracts summaries from existing descriptions |
| **Testable in isolation** | Handler class can be unit tested independently |

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/strands/tools/lazy_tool_handler.py` | **NEW** | ~150 |
| `src/strands/agent/agent.py` | Add parameter + register | ~5 |
| `src/strands/event_loop/event_loop.py` | Single transform call | ~3 |
| `src/strands/tools/executors/_executor.py` | Optional blocking | ~5 |

**Total: ~163 lines, mostly in one new file**

---

## Open Questions

1. **Block vs warn on unexpanded calls?**
   - Configurable via `require_expansion` parameter

2. **Persist expanded state across invocations?**
   - Currently per-invocation; could add `persistent=True` option

3. **Support `expand_tool("*")` for expand-all?**
   - Easy to add in the handler's `expand()` method
