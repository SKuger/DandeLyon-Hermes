"""The same tools, over the Model Context Protocol.

The bot is one consumer of these tools. A support engineer with Claude
Desktop open is another, and there is no reason for them to be a
different implementation: "what is our returns policy" is the same
question whether it arrives on WhatsApp or from someone's laptop.

So this is a thin adapter, not a second tool layer. `ToolRegistry`
already produces JSON Schema and already turns expected failures into
`ToolError`, which is exactly the shape MCP wants: a schema per tool and
a result flagged `is_error` rather than an exception. The whole server
is two handlers.

`escalate_to_human` is deliberately absent. It is bound to a
conversation, and an MCP client does not have one; exposing it would
mean inventing a conversation id for a side effect that pages a person.
The read-only tools are the ones that generalise.

Optional. The MCP SDK is not in `requirements.txt`:

    pip install -r requirements-mcp.txt

Run it over stdio:

    python -m app.mcp_server

Or point a client at it, e.g. in Claude Desktop's config:

    {"mcpServers": {"hermes": {"command": "python",
                               "args": ["-m", "app.mcp_server"]}}}
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.agent.loop import run_tool
from app.agent.tools import CurrentTime, SearchKnowledgeBase, ToolRegistry
from app.config import settings
from app.rag.index import LexicalIndex, load_directory

logger = logging.getLogger(__name__)

SERVER_NAME = "dandelyon-hermes"
INSTRUCTIONS = (
    "Tools from a customer support bot: the current time, and search over "
    "the support knowledge base. The search returns passages with their "
    "source, or tells you plainly that nothing matched — when it does, say "
    "you do not know rather than answering from memory."
)


def build_server(registry: ToolRegistry):
    """Wrap a registry as an MCP server.

    Takes the registry rather than building one so a caller can decide
    which tools to expose. Nothing here knows what a tool does.
    """
    from mcp.server import Server
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    async def on_list_tools(_context, _params) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=schema["name"],
                    description=schema["description"],
                    input_schema=schema["input_schema"],
                )
                for schema in registry.schemas()
            ]
        )

    async def on_call_tool(_context, params) -> CallToolResult:
        # `run_tool` is the same function the agent loop uses, so a
        # timeout or a refused call means the same thing to a desktop
        # client as it does to someone on WhatsApp.
        content, is_error = await run_tool(
            registry, params.name, dict(params.arguments or {})
        )
        return CallToolResult(
            content=[TextContent(type="text", text=content)],
            is_error=is_error,
        )

    return Server(
        SERVER_NAME,
        version="1.2.0",
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def build_registry() -> ToolRegistry:
    """The conversation-independent tools, with the corpus loaded."""
    index = LexicalIndex()
    directory = Path(settings.knowledge_dir)
    if directory.is_dir():
        count = await load_directory(index, directory)
        logger.info("knowledge base: %d chunks from %s", count, directory)
    else:
        logger.warning("no knowledge directory at %s", directory)

    return ToolRegistry([CurrentTime(), SearchKnowledgeBase(index)])


async def serve() -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(await build_registry())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    import asyncio

    # stdout is the transport. Anything printed there corrupts the
    # protocol, so logging goes to stderr.
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
