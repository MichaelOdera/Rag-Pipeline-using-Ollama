import asyncio
import json
import os
from typing import Any, Dict, Optional

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - optional dependency
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


def is_mcp_enabled() -> bool:
    value = os.getenv("MCP_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_mcp_tool_config() -> Optional[Dict[str, Any]]:
    if not is_mcp_enabled():
        return None

    command = os.getenv("MCP_SERVER_COMMAND", "").strip()
    if not command:
        return None

    args = []
    raw_args = os.getenv("MCP_SERVER_ARGS", "").strip()
    if raw_args:
        args = [item.strip() for item in raw_args.split(",") if item.strip()]

    tool_name = os.getenv("MCP_TOOL_NAME", "").strip() or "get_context"

    tool_args: Dict[str, Any] = {}
    raw_tool_args = os.getenv("MCP_TOOL_ARGS_JSON", "").strip()
    if raw_tool_args:
        try:
            tool_args = json.loads(raw_tool_args)
        except json.JSONDecodeError:
            tool_args = {}

    return {
        "command": command,
        "args": args,
        "tool_name": tool_name,
        "tool_args": tool_args,
    }


def _extract_text_from_result(result: Any) -> str:
    if result is None:
        return ""

    if hasattr(result, "content") and result.content:
        chunks = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        if chunks:
            return "\n".join(chunks)

    if hasattr(result, "text") and result.text:
        return result.text

    return str(result)


async def _invoke_mcp_tool(question: str) -> Optional[str]:
    if ClientSession is None or StdioServerParameters is None or stdio_client is None:
        return None

    config = get_mcp_tool_config()
    if not config:
        return None

    tool_args = dict(config["tool_args"])
    tool_args.setdefault("question", question)

    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(config["tool_name"], tool_args)
            return _extract_text_from_result(result)


def fetch_mcp_context(question: str) -> Optional[str]:
    if not is_mcp_enabled():
        return None

    try:
        return asyncio.run(_invoke_mcp_tool(question))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_invoke_mcp_tool(question))
        finally:
            loop.close()


def build_context_with_mcp(retrieved_context: str, mcp_context: Optional[str]) -> str:
    if not mcp_context:
        return retrieved_context

    cleaned_context = mcp_context.strip()
    if not cleaned_context:
        return retrieved_context

    return "\n\n".join(
        [retrieved_context.strip(), f"Additional MCP context:\n{cleaned_context}"]
    )
