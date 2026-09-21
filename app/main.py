"""HTTP layer.

The one rule that shapes this file: Meta expects a 200 within seconds,
and retries the whole webhook when it does not get one. A model call
takes longer than that. So the endpoint does the cheap part — verify,
parse, acknowledge — and hands the slow part to a background task.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request, Response

from app.agent.llm import ScriptedToolProvider
from app.agent.loop import AgentLoop
from app.agent.tools import (
    CurrentTime,
    EscalateToHuman,
    SearchKnowledgeBase,
    ToolRegistry,
)
from app.channels.meta import ADAPTERS
from app.config import settings
from app.core.models import Channel
from app.core.pipeline import Pipeline
from app.core.security import is_valid_signature
from app.core.sender import GraphApiSender, LoggingSender
from app.core.store import InMemoryStore
from app.rag.embeddings import VoyageEmbedder
from app.rag.index import DenseIndex, LexicalIndex, VectorIndex, load_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index() -> VectorIndex:
    """Lexical by default; dense when an embedding key is configured.

    Lexical matches on shared vocabulary, which covers most support
    questions and costs nothing. It cannot bridge a synonym, which is
    what the key buys.
    """
    if settings.voyage_api_key:
        return DenseIndex(VoyageEmbedder(settings.voyage_api_key))
    return LexicalIndex()


knowledge = build_index()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Index the knowledge base once, at boot.

    Reading and embedding files on every question would make the first
    tool call slower than the model call it was supposed to ground.
    """
    directory = Path(settings.knowledge_dir)
    if directory.is_dir():
        count = await load_directory(knowledge, directory)
        logger.info("knowledge base: %d chunks from %s", count, directory)
    else:
        logger.warning(
            "no knowledge directory at %s: search will find nothing",
            directory,
        )
    yield


app = FastAPI(
    title="DandeLyon Hermes", version="1.1.0", lifespan=lifespan
)


def build_pipeline() -> Pipeline:
    """Wire the implementations chosen by the environment.

    Every dependency has a no-credentials default, which is what lets
    `docker compose up` work on a machine that has never seen a Meta app.
    """
    store = InMemoryStore()
    if settings.redis_url:
        import redis.asyncio as aioredis

        from app.core.store import RedisStore

        store = RedisStore(aioredis.from_url(settings.redis_url))

    provider = ScriptedToolProvider()
    if settings.anthropic_api_key:
        from app.ai.anthropic_agent import AnthropicAgentProvider

        provider = AnthropicAgentProvider(
            settings.anthropic_api_key, settings.model
        )

    def tools_for(message) -> ToolRegistry:
        # Rebuilt per message: `escalate_to_human` has to escalate this
        # conversation, so the tool is bound to it rather than reading a
        # global.
        return ToolRegistry(
            [
                CurrentTime(),
                SearchKnowledgeBase(knowledge),
                EscalateToHuman(store, message.conversation_id),
            ]
        )

    sender = LoggingSender()
    if settings.graph_token:
        sender = GraphApiSender(
            settings.graph_token,
            settings.phone_number_id,
            settings.page_id,
        )

    responder = AgentLoop(provider=provider, tools_for=tools_for)
    return Pipeline(store=store, responder=responder, sender=sender)


pipeline = build_pipeline()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/webhook/{channel}")
async def verify(channel: Channel, request: Request) -> Response:
    """Meta's subscription handshake: echo the challenge back."""
    params = request.query_params
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == settings.verify_token
    ):
        return Response(content=params.get("hub.challenge", ""))
    return Response(status_code=403)


@app.post("/webhook/{channel}")
async def receive(
    channel: Channel, request: Request, background: BackgroundTasks
) -> Response:
    body = await request.body()

    if settings.verify_signatures and not is_valid_signature(
        body, request.headers.get("X-Hub-Signature-256"), settings.app_secret
    ):
        return Response(status_code=403)

    payload = await request.json()
    messages = ADAPTERS[channel].parse(payload)

    for message in messages:
        background.add_task(_process, message)

    # Answered before a single message has been handled. This is the
    # whole point: the 200 is not a claim that the work is done, it is an
    # acknowledgement that the payload was received and will be handled.
    return Response(status_code=200)


async def _process(message) -> None:
    try:
        await pipeline.handle(message)
    except Exception:  # noqa: BLE001
        # Nothing above can catch this: the response is already sent.
        # An unhandled error here would disappear silently.
        logger.exception("failed to handle %s", message.message_id)
