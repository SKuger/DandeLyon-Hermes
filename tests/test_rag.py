"""Retrieval: what it finds, what it refuses to find, and what it cannot.

The last one matters most. A retriever that always returns something is
a hallucination generator with extra steps.
"""

from pathlib import Path

import pytest

from app.agent.llm import ScriptedToolProvider
from app.agent.loop import AgentLoop
from app.agent.tools import NO_MATCH, SearchKnowledgeBase, ToolError, ToolRegistry
from app.core.models import Channel, InboundMessage
from app.rag.index import (
    CHUNK_CHARS,
    LexicalIndex,
    chunk_text,
    load_directory,
    tokenize,
)

KNOWLEDGE = Path(__file__).resolve().parent.parent / "knowledge"


def make_message(text: str) -> InboundMessage:
    return InboundMessage(
        message_id="m1",
        conversation_id=InboundMessage.build_conversation_id(
            Channel.WHATSAPP, "573001112233"
        ),
        channel=Channel.WHATSAPP,
        sender_id="573001112233",
        text=text,
    )


@pytest.fixture
async def index() -> LexicalIndex:
    built = LexicalIndex()
    await load_directory(built, KNOWLEDGE)
    return built


# --- chunking ------------------------------------------------------------


def test_chunks_stay_within_budget_and_keep_their_source():
    text = "\n\n".join(f"Paragraph {n}. " + "word " * 40 for n in range(12))

    chunks = chunk_text(text, "manual.md")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= CHUNK_CHARS + 200 for chunk in chunks)
    assert all(chunk.source == "manual.md" for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_a_short_document_is_one_chunk():
    assert len(chunk_text("Just one line.", "note.md")) == 1


def test_stopwords_are_dropped_so_short_questions_are_not_mostly_noise():
    assert tokenize("How do I get a refund?") == ["get", "refund", "get_refund"]


# --- search --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_question_finds_its_document(index):
    hits = await index.search("how long does shipping take?")

    assert hits
    assert hits[0][0].source == "shipping.md"


@pytest.mark.asyncio
async def test_results_carry_a_citation(index):
    (chunk, _score), *_ = await index.search("refund on an opened item")

    assert chunk.citation.startswith("returns.md#")


@pytest.mark.asyncio
async def test_an_unrelated_question_returns_nothing(index):
    """The whole point of the threshold."""
    assert await index.search("what is the capital of France?") == []
    assert await index.search("do you sell bicycles?") == []


@pytest.mark.asyncio
async def test_an_empty_index_returns_nothing():
    assert await LexicalIndex().search("anything") == []


@pytest.mark.asyncio
async def test_lexical_search_cannot_bridge_a_synonym(index):
    """A known limitation, asserted so it is a decision and not a bug.

    The corpus covers this: "A damaged item is not a return: send a
    photo and a replacement is shipped immediately." Nothing in the
    question shares vocabulary with it, so a lexical index cannot reach
    it and correctly says so rather than guessing. Bridging this is what
    `DenseIndex` and an embedding model are for.
    """
    assert await index.search("my package arrived broken") == []


# --- the tool ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tool_returns_passages_with_sources(index):
    result = await SearchKnowledgeBase(index).run(
        query="how long does shipping take?"
    )

    assert "shipping.md#" in result
    assert "business days" in result


@pytest.mark.asyncio
async def test_a_miss_tells_the_model_to_admit_it_does_not_know(index):
    """The model needs an instruction here, not an empty string."""
    result = await SearchKnowledgeBase(index).run(
        query="what is the capital of France?"
    )

    assert "Nothing in the knowledge base" in result
    assert "pass them to a person" in result


@pytest.mark.asyncio
async def test_an_empty_query_is_a_tool_error(index):
    with pytest.raises(ToolError):
        await SearchKnowledgeBase(index).run(query="   ")


# --- through the loop ----------------------------------------------------


@pytest.mark.asyncio
async def test_an_internal_instruction_never_reaches_the_user(index):
    """`NO_MATCH` is addressed to the model. A customer must never read it.

    The first version of this shipped the instruction straight to the
    user's phone, because the scripted provider repeated tool output
    verbatim. A real model would have rewritten it; a stand-in will not.
    """
    loop = AgentLoop(
        provider=ScriptedToolProvider(),
        tools_for=lambda _: ToolRegistry([SearchKnowledgeBase(index)]),
    )

    reply = await loop.respond(
        make_message("what is the capital of France?"), []
    )

    assert NO_MATCH not in reply
    assert "pass you to a person" in reply


@pytest.mark.asyncio
async def test_a_hit_comes_back_as_prose_without_the_citation_header(index):
    loop = AgentLoop(
        provider=ScriptedToolProvider(),
        tools_for=lambda _: ToolRegistry([SearchKnowledgeBase(index)]),
    )

    reply = await loop.respond(
        make_message("how long does shipping take?"), []
    )

    assert "business days" in reply
    assert "score" not in reply
    assert not reply.startswith("[")
