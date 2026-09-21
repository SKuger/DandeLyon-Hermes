"""The MCP adapter, exercised through a real client over the protocol.

Calling the handlers directly would test the functions and not the
server. These tests speak MCP to it: initialize, list, call.

Skipped when the SDK is not installed — it is an optional dependency.
"""

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.shared.memory import (  # noqa: E402
    create_client_server_memory_streams,
)

from app.agent.tools import CurrentTime, SearchKnowledgeBase, ToolRegistry  # noqa: E402
from app.mcp_server import build_server  # noqa: E402
from app.rag.index import LexicalIndex, load_directory  # noqa: E402
from tests.test_rag import KNOWLEDGE  # noqa: E402


async def registry() -> ToolRegistry:
    index = LexicalIndex()
    await load_directory(index, KNOWLEDGE)
    return ToolRegistry([CurrentTime(), SearchKnowledgeBase(index)])


class connected:
    """A client session talking to the server over in-memory streams."""

    def __init__(self, server) -> None:
        self._server = server

    async def __aenter__(self) -> ClientSession:
        self._streams = create_client_server_memory_streams()
        (client_read, client_write), (server_read, server_write) = (
            await self._streams.__aenter__()
        )
        self._tasks = anyio.create_task_group()
        await self._tasks.__aenter__()
        self._tasks.start_soon(
            lambda: self._server.run(
                server_read,
                server_write,
                self._server.create_initialization_options(),
            )
        )
        self._session = ClientSession(client_read, client_write)
        session = await self._session.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, *exc) -> None:
        await self._session.__aexit__(*exc)
        self._tasks.cancel_scope.cancel()
        await self._tasks.__aexit__(None, None, None)
        await self._streams.__aexit__(None, None, None)


def text_of(result) -> str:
    return "".join(
        block.text for block in result.content if block.type == "text"
    )


@pytest.mark.asyncio
async def test_the_registry_is_published_as_mcp_tools():
    async with connected(build_server(await registry())) as session:
        listed = await session.list_tools()

    names = {tool.name for tool in listed.tools}
    assert names == {"current_time", "search_knowledge_base"}

    search = next(t for t in listed.tools if t.name == "search_knowledge_base")
    # The schema crossed the protocol unchanged: tools are written once.
    assert search.input_schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_escalation_is_not_exposed():
    """It is bound to a conversation an MCP client does not have."""
    async with connected(build_server(await registry())) as session:
        listed = await session.list_tools()

    assert "escalate_to_human" not in {tool.name for tool in listed.tools}


@pytest.mark.asyncio
async def test_a_client_can_search_the_knowledge_base():
    async with connected(build_server(await registry())) as session:
        result = await session.call_tool(
            "search_knowledge_base",
            {"query": "how long does shipping take?"},
        )

    assert not result.is_error
    assert "business days" in text_of(result)
    assert "shipping.md#" in text_of(result)


@pytest.mark.asyncio
async def test_a_miss_is_reported_as_text_not_as_a_failure():
    """Finding nothing is an answer. `is_error` is for broken calls."""
    async with connected(build_server(await registry())) as session:
        result = await session.call_tool(
            "search_knowledge_base", {"query": "capital of France"}
        )

    assert not result.is_error
    assert "Nothing in the knowledge base" in text_of(result)


@pytest.mark.asyncio
async def test_a_bad_call_comes_back_flagged_rather_than_crashing():
    """`ToolError` becomes `is_error`, which is what MCP expects."""
    async with connected(build_server(await registry())) as session:
        missing = await session.call_tool("search_knowledge_base", {})
        unknown = await session.call_tool("send_refund", {})

    assert missing.is_error
    assert unknown.is_error
    assert "current_time" in text_of(unknown)  # it lists what does exist
