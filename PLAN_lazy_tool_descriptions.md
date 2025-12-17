# Implementation Plan: Lazy Tool Description Loading

## Problem Statement

When an agent has many tools, sending full descriptions for all tools in every API call:
- Consumes significant context window space
- Increases token costs
- May hit context limits with large tool sets

## Proposed Solution: Progressive Tool Description Expansion

Provide abbreviated tool descriptions initially, allowing the model to request full descriptions for tools it intends to use. This trades an extra API call for significant context savings.

---

## Design Options

### Option A: Meta-Tool Pattern (Recommended)

Add a special `expand_tool` meta-tool that the model calls to get full descriptions.

**Flow:**
```
1. Agent sends: [tool_name + summary] for all tools + expand_tool
2. Model decides it needs "complex_tool"
3. Model calls: expand_tool(name="complex_tool")
4. System returns: full description + schema details
5. Model calls: complex_tool(params...)
```

**Pros:**
- Clean, explicit pattern
- Model controls when to expand
- Works with existing tool infrastructure
- No changes to model providers

**Cons:**
- Adds 1 API round-trip per tool expansion
- Model must learn to use the meta-tool

---

### Option B: Automatic Expansion on First Use

Intercept tool calls, check if model has seen full description, expand if needed.

**Flow:**
```
1. Agent sends: [tool_name + summary] for all tools
2. Model calls: complex_tool(params...)
3. System intercepts, sees tool not expanded
4. System re-prompts with full description
5. Model confirms/adjusts call
6. Tool executes
```

**Pros:**
- Transparent to model
- No meta-tool needed

**Cons:**
- May cause incorrect tool usage on first attempt
- More complex state management
- Potential for wasted API calls if params were wrong

---

### Option C: Hybrid - Summary with On-Demand Schema

Send descriptions but defer parameter schemas until needed.

**Flow:**
```
1. Agent sends: [tool_name + description] (no inputSchema)
2. Model indicates intent: "I want to use complex_tool"
3. System provides: full inputSchema
4. Model calls with correct params
```

**Pros:**
- Schemas are often the largest part of tool definitions
- Description helps model decide relevance

**Cons:**
- Requires model behavior change
- Less context savings than full lazy loading

---

## Recommended Approach: Option A (Meta-Tool Pattern)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              ToolRegistry                            │    │
│  │  ┌──────────────┐  ┌──────────────────────────┐     │    │
│  │  │ LazyToolSpec │  │ ExpandedToolCache        │     │    │
│  │  │ - name       │  │ - expanded_tools: set    │     │    │
│  │  │ - summary    │  │ - get_full_spec()        │     │    │
│  │  │ - full_spec  │  └──────────────────────────┘     │    │
│  │  └──────────────┘                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           ExpandToolMetaTool                         │    │
│  │  - Returns full description for requested tool       │    │
│  │  - Marks tool as "expanded" in cache                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Data Structures

#### 1.1 Extend ToolSpec Type
**File:** `src/strands/types/tools.py`

```python
class LazyToolSpec(TypedDict, total=False):
    """Tool specification with lazy description loading."""
    name: str
    summary: str  # Short description (1-2 sentences)
    description: str  # Full description (only populated when expanded)
    inputSchema: InputSchema
    outputSchema: JSONSchema

class ToolDescriptionMode(Enum):
    """How tool descriptions are provided to the model."""
    FULL = "full"           # All descriptions sent upfront (current behavior)
    LAZY = "lazy"           # Summary only, expand on demand
    HYBRID = "hybrid"       # Description without schema until needed
```

#### 1.2 Add Summary Field to Tool Decorator
**File:** `src/strands/tools/decorator.py`

```python
@tool(
    name="my_tool",
    summary="Brief one-liner for lazy mode",  # NEW
    description="Full detailed description..."
)
def my_tool(...):
    ...
```

- Extract first sentence of docstring as default summary
- Allow explicit `summary` parameter override

---

### Phase 2: Meta-Tool Implementation

#### 2.1 Create ExpandTool
**File:** `src/strands/tools/expand_tool.py` (new file)

```python
@tool(
    name="expand_tool",
    description="Get the full description and parameter schema for a tool before using it. "
                "Call this when you need more details about how to use a specific tool."
)
def expand_tool(
    tool_name: str,
    *,
    _agent: "Agent",  # Injected
    _invocation_state: dict,  # Injected
) -> str:
    """
    Expand a tool's description to see its full details.

    Args:
        tool_name: The name of the tool to expand

    Returns:
        Full tool description and parameter schema
    """
    registry = _agent.tool_registry

    if tool_name not in registry:
        return f"Error: Tool '{tool_name}' not found"

    full_spec = registry.get_full_tool_spec(tool_name)

    # Mark as expanded in invocation state
    expanded = _invocation_state.setdefault("_expanded_tools", set())
    expanded.add(tool_name)

    return format_tool_spec_for_model(full_spec)
```

#### 2.2 Format Helper
```python
def format_tool_spec_for_model(spec: ToolSpec) -> str:
    """Format a tool spec as human-readable text for the model."""
    lines = [
        f"## Tool: {spec['name']}",
        f"\n### Description\n{spec['description']}",
        f"\n### Parameters\n```json\n{json.dumps(spec['inputSchema'], indent=2)}\n```"
    ]
    if spec.get('outputSchema'):
        lines.append(f"\n### Output Schema\n```json\n{json.dumps(spec['outputSchema'], indent=2)}\n```")
    return "\n".join(lines)
```

---

### Phase 3: Registry Modifications

#### 3.1 Lazy Spec Generation
**File:** `src/strands/tools/registry.py`

```python
class ToolRegistry:
    def __init__(self, ..., description_mode: ToolDescriptionMode = ToolDescriptionMode.FULL):
        self._description_mode = description_mode
        self._expanded_tools: set[str] = set()

    def get_tool_specs_for_model(
        self,
        invocation_state: dict | None = None
    ) -> list[ToolSpec]:
        """Get tool specs formatted according to description mode."""

        if self._description_mode == ToolDescriptionMode.FULL:
            return self.get_all_tool_specs()  # Current behavior

        expanded = set()
        if invocation_state:
            expanded = invocation_state.get("_expanded_tools", set())

        specs = []
        for name, tool in self._tools.items():
            if name in expanded:
                # Tool was expanded, send full spec
                specs.append(tool.tool_spec)
            else:
                # Send lazy spec with summary only
                specs.append(self._create_lazy_spec(tool))

        # Always include expand_tool with full spec
        if self._description_mode == ToolDescriptionMode.LAZY:
            specs.append(self._expand_tool.tool_spec)

        return specs

    def _create_lazy_spec(self, tool: AgentTool) -> ToolSpec:
        """Create a minimal spec with just name and summary."""
        full_spec = tool.tool_spec
        return {
            "name": full_spec["name"],
            "description": self._get_summary(tool),
            "inputSchema": {"json": {"type": "object", "properties": {}}}
        }

    def _get_summary(self, tool: AgentTool) -> str:
        """Get or generate a summary for the tool."""
        # Check for explicit summary
        if hasattr(tool, 'summary') and tool.summary:
            return tool.summary

        # Extract first sentence from description
        desc = tool.tool_spec.get("description", "")
        first_sentence = desc.split(". ")[0]
        return first_sentence + "." if first_sentence else "No description available."
```

---

### Phase 4: Agent Configuration

#### 4.1 Agent Constructor Update
**File:** `src/strands/agent/agent.py`

```python
class Agent:
    def __init__(
        self,
        ...,
        tool_description_mode: ToolDescriptionMode | str = ToolDescriptionMode.FULL,
    ):
        # Convert string to enum if needed
        if isinstance(tool_description_mode, str):
            tool_description_mode = ToolDescriptionMode(tool_description_mode)

        self._tool_description_mode = tool_description_mode

        # Auto-register expand_tool if in lazy mode
        if tool_description_mode == ToolDescriptionMode.LAZY:
            self.tool_registry.register_tool(expand_tool)
```

#### 4.2 Event Loop Integration
**File:** `src/strands/event_loop/event_loop.py`

```python
# Line ~339: Change from:
tool_specs = agent.tool_registry.get_all_tool_specs()

# To:
tool_specs = agent.tool_registry.get_tool_specs_for_model(invocation_state)
```

---

### Phase 5: Tool Validation Handling

When in lazy mode, the model might try to call a tool without expanding it first. Handle gracefully:

#### 5.1 Pre-execution Check
**File:** `src/strands/tools/executors/_executor.py`

```python
@staticmethod
async def _stream(...):
    # Check if tool needs expansion first
    if agent._tool_description_mode == ToolDescriptionMode.LAZY:
        expanded = invocation_state.get("_expanded_tools", set())
        if tool_name not in expanded and tool_name != "expand_tool":
            # Return helpful message instead of executing
            yield ToolResultEvent(
                tool_use_id=tool_use["toolUseId"],
                status="error",
                content=f"Please use expand_tool('{tool_name}') first to see the full "
                        f"description and parameters before calling this tool."
            )
            return

    # Continue with normal execution...
```

---

### Phase 6: Summary Extraction Enhancement

#### 6.1 Automatic Summary Generation
**File:** `src/strands/tools/decorator.py`

```python
def _extract_summary_from_docstring(docstring: str | None) -> str:
    """Extract a concise summary from a docstring."""
    if not docstring:
        return "No description available."

    # Get first paragraph (up to blank line)
    paragraphs = docstring.strip().split("\n\n")
    first_para = paragraphs[0].replace("\n", " ").strip()

    # Limit to ~100 chars for summary
    if len(first_para) > 100:
        # Find last complete sentence within limit
        sentences = first_para.split(". ")
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < 100:
                summary += sentence + ". "
            else:
                break
        return summary.strip() or first_para[:97] + "..."

    return first_para
```

---

## Usage Example

```python
from strands import Agent
from strands.types.tools import ToolDescriptionMode

# Create agent with lazy tool descriptions
agent = Agent(
    model=my_model,
    tools=[tool1, tool2, tool3, ...many_tools...],
    tool_description_mode=ToolDescriptionMode.LAZY
)

# The agent will now:
# 1. Send summaries for all tools
# 2. Model calls expand_tool("tool2") when it needs details
# 3. Model then calls tool2 with correct parameters
result = agent("Perform a complex task requiring tool2")
```

---

## Context Savings Analysis

**Example with 20 tools:**

| Mode | Tokens per tool | Total tokens |
|------|-----------------|--------------|
| FULL | ~200 (desc + schema) | ~4,000 |
| LAZY (summary) | ~30 | ~600 |
| LAZY (after 3 expansions) | ~30×17 + ~200×3 | ~1,110 |

**Savings:** 70-85% context reduction for typical usage patterns.

---

## Testing Strategy

### Unit Tests
1. `test_lazy_spec_generation` - Verify summary extraction
2. `test_expand_tool_returns_full_spec` - Meta-tool functionality
3. `test_unexpanded_tool_blocked` - Validation behavior
4. `test_expanded_tools_tracked` - State management

### Integration Tests
1. `test_lazy_mode_end_to_end` - Full agent flow
2. `test_model_learns_to_expand` - Model behavior with lazy tools
3. `test_mixed_expanded_unexpanded` - Partial expansion state

### Performance Tests
1. `test_context_savings` - Measure actual token reduction
2. `test_api_call_overhead` - Measure expansion round-trip cost

---

## Migration & Backwards Compatibility

- Default `tool_description_mode=FULL` preserves current behavior
- No changes required for existing code
- Opt-in via configuration parameter

---

## Future Enhancements

1. **Smart Expansion Hints:** System suggests tools to expand based on task
2. **Persistent Expansion Cache:** Remember expanded tools across invocations
3. **Category-based Expansion:** "Expand all file tools"
4. **Usage-based Caching:** Auto-expand frequently used tools

---

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `src/strands/types/tools.py` | Modify | Add LazyToolSpec, ToolDescriptionMode |
| `src/strands/tools/expand_tool.py` | Create | Meta-tool implementation |
| `src/strands/tools/decorator.py` | Modify | Add summary extraction, summary param |
| `src/strands/tools/registry.py` | Modify | Add lazy spec generation |
| `src/strands/agent/agent.py` | Modify | Add tool_description_mode param |
| `src/strands/event_loop/event_loop.py` | Modify | Use new spec method |
| `src/strands/tools/executors/_executor.py` | Modify | Add expansion validation |
| `tests/test_lazy_tools.py` | Create | Test suite |

---

## Open Questions for Discussion

1. **Should expand_tool count against tool call limits?**
   - Probably not, since it's infrastructure

2. **What if model ignores expand_tool and guesses parameters?**
   - Option: Block execution (recommended for correctness)
   - Option: Allow with warning (more flexible)

3. **Should we support "expand all" for small tool sets?**
   - Could add `expand_tool(name="*")` pattern

4. **How to handle MCP tools?**
   - MCP tools have their own description mechanism
   - May need adapter layer

5. **Should expanded state persist across conversation turns?**
   - Yes, within same invocation
   - Configurable for multi-turn conversations
