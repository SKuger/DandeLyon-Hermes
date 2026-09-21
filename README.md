# DandeLyon Hermes

[![CI](https://github.com/SKuger/DandeLyon-Hermes/actions/workflows/ci.yml/badge.svg)](https://github.com/SKuger/DandeLyon-Hermes/actions/workflows/ci.yml)

**Omnichannel messaging bot: WhatsApp, Instagram and Messenger.**

A FastAPI webhook service that receives messages from the three Meta
channels, normalizes them into one internal model, answers with a
tool-using AI agent, and sends the reply back through the channel it came
from.

It runs with no credentials: `docker compose up` gives you a working
service — including a working agent loop — backed by a scripted provider
and an in-memory store. Add a Meta app and an Anthropic key and the same
code talks to the real thing.

```
Meta webhook ──► FastAPI ──► signature check ──► channel adapter
                    │                                  │
                    └── 200 OK (immediately)           ▼
                                              ┌─── InboundMessage ───┐
                                              │  dedupe · handoff    │
                                              │  agent loop · tools  │
                                              │  history · send      │
                                              └──────────────────────┘
                                                   (background)
```

## The six problems this solves

**1. Three channels, three payload shapes.**
WhatsApp nests messages under `entry[].changes[].value.messages[]` with the
body at `text.body`. Messenger and Instagram use `entry[].messaging[]` with
the body at `text`. Each channel gets an adapter
([`app/channels/meta.py`](app/channels/meta.py)); everything above them only
ever sees `InboundMessage`. Adding a fourth channel is one new file.

**2. The timeout.**
Meta expects a 200 within seconds and retries the whole webhook when it does
not get one. A model call takes longer than that. So the endpoint verifies,
parses and acknowledges — then hands the work to a background task
([`app/main.py`](app/main.py)). The 200 means *received*, not *done*.

**3. Retries turn into duplicate replies.**
Because providers retry, the same message arrives two or three times, and a
bot that answers twice is instantly obvious. Every message id is claimed
before any work starts ([`app/core/store.py`](app/core/store.py)). On Redis
the claim is a `SET NX EX`, so two replicas receiving the same retry at the
same moment cannot both win it.

**4. The model is not always there.**
It times out, it rate-limits, it fails. The AI layer sits behind a Protocol
with a bounded timeout and a fallback message
([`app/ai/`](app/ai)), because silence on the user's phone is worse than an
honest "a person will follow up".

**5. A model that acts, not just answers.**
The moment the model can call a tool, its mistakes stop being embarrassing
and start being expensive: a bad sentence is a bad sentence, a bad tool call
opens two support tickets. The agent loop
([`app/agent/loop.py`](app/agent/loop.py)) is a dozen lines; the constraints
around it are the point.

- **The step budget is the bill, not a safety net.** Every iteration is
  another model call with a longer prompt, so the ceiling is explicit, low,
  and logged as an incident when it is hit.
- **A tool that hangs is worse than a tool that fails.** Each call gets its
  own timeout, shorter than the model's, and a timeout comes back as an
  observation the model can act on.
- **A failed tool is information.** `ToolError` becomes text the model
  reads, because "that record does not exist" is often the answer the user
  needed.
- **Side effects are idempotent.** `escalate_to_human` is safe to call
  twice: on Redis the handoff is claimed with the same `SET NX` used for
  deduplication, because a model repeats itself and a webhook arrives twice.
- **After a handoff the bot goes quiet.** Two voices answering the same
  customer is worse than a slow reply, and it is the one state the model is
  not allowed to overrule ([`app/core/pipeline.py`](app/core/pipeline.py)).

Tools declare their arguments as JSON Schema and implement them as named
parameters, so a hallucinated argument is caught by Python's own binding
instead of a hand-written check that would drift from the schema.

**6. A model that does not know, and says so.**
Grounding is what stops a support bot inventing a returns policy. The
retrieval here ([`app/rag/`](app/rag)) is ordinary — chunk with overlap,
score, take the top few — except for the part that usually gets skipped: a
**score threshold**, below which it returns *nothing*. A retriever that
always hands back its top three results will hand the model three
irrelevant paragraphs, and the model will build an answer out of them. On
a miss the tool instead returns an instruction to admit ignorance and offer
a person.

Every passage carries a citation (`returns.md#0`), because an answer nobody
can trace back to a document is not much better than a guess.

The default index is **lexical**: TF-IDF over the corpus, no model, no key,
no network. IDF is what makes it work — without it a question's common
words outweigh the one term that carries its meaning, and the top hit is
whichever chunk happens to be longest. It has a real limitation, and there
is a test asserting it rather than a comment hoping you do not notice: it
cannot match "my package arrived broken" against a document that says
"damaged item". Bridging vocabulary is what an embedding model buys, and
`DenseIndex` takes one when `VOYAGE_API_KEY` is set.

**On frameworks.** There is no LangChain in the default path, deliberately.
This project's contract is that it runs with no key and no network, and the
loop is the part worth reading — hiding it inside a framework would cost
both.

That is a claim about a boundary, so the boundary is exercised rather than
asserted: [`app/agent/langgraph_loop.py`](app/agent/langgraph_loop.py) is the
same agent as a LangGraph `StateGraph`, satisfying the same `Responder`
contract, driving the same `ToolRegistry`, and sharing the same `run_tool`
so a timeout means the same thing in both. What changes is who owns the
control flow — a `for` loop, or a graph where the budget becomes a recursion
limit. What does not change is the tools: writing them twice, once per
framework, would have been the sign the boundary was decorative.

```bash
pip install -r requirements-langgraph.txt
```

## The tools are not only the bot's

The bot is one consumer of these tools. A support engineer with an MCP
client open is another, and "what is our returns policy" is the same
question whether it arrives on WhatsApp or from someone's laptop — so
[`app/mcp_server.py`](app/mcp_server.py) publishes the same registry over
the Model Context Protocol. It is two handlers: `ToolRegistry` already
emits JSON Schema and already turns expected failures into `ToolError`,
which is the shape MCP wants.

`escalate_to_human` is deliberately not published. It is bound to a
conversation an MCP client does not have, and exposing it would mean
inventing one for a side effect that pages a person.

```bash
pip install -r requirements-mcp.txt
python -m app.mcp_server
```

```json
{"mcpServers": {"hermes": {"command": "python",
                           "args": ["-m", "app.mcp_server"]}}}
```

## Security

The webhook URL is public, so the signature is the only thing separating a
real message from a forged one. `X-Hub-Signature-256` is verified against
the **raw** request body with `hmac.compare_digest`
([`app/core/security.py`](app/core/security.py)) — raw, because
re-serializing parsed JSON changes the bytes and breaks the signature;
`compare_digest`, because `==` leaks how many leading characters matched.

With no `META_APP_SECRET` set, verification is skipped so the project can be
run locally. That is the only place the check relaxes, and it is explicit
about it in [`app/config.py`](app/config.py).

## Run it

```bash
cp .env.example .env
docker compose up --build
```

```bash
curl localhost:8000/health

curl -X POST localhost:8000/webhook/whatsapp \
  -H 'Content-Type: application/json' \
  -d '{"entry":[{"changes":[{"value":{"messages":[
       {"id":"wamid.1","from":"573001112233","type":"text",
        "text":{"body":"hello"}}]}}]}]}'
```

The reply appears in the service logs (`docker compose logs -f api`). Send
the same payload twice and only one reply is produced — that is the
idempotency guard doing its job.

Ask something the knowledge base covers and the agent looks it up first:

```bash
curl -X POST localhost:8000/webhook/whatsapp \
  -H 'Content-Type: application/json' \
  -d '{"entry":[{"changes":[{"value":{"messages":[
       {"id":"wamid.3","from":"573001112233","type":"text",
        "text":{"body":"how long does shipping take?"}}]}}]}]}'
```

Ask something it does not cover — `what is the capital of France?` — and the
reply is "I do not have that information", not an invented one.

To watch the agent take an action rather than answer, ask for a person:

```bash
curl -X POST localhost:8000/webhook/whatsapp \
  -H 'Content-Type: application/json' \
  -d '{"entry":[{"changes":[{"value":{"messages":[
       {"id":"wamid.2","from":"573001112233","type":"text",
        "text":{"body":"I want to talk to a person"}}]}}]}]}'
```

The logs show the loop calling `escalate_to_human`, the handoff opening,
and the reply going out. Send anything else from that number afterwards and
the bot stays silent: the conversation now belongs to a human.

Without Docker:

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

Covers channel normalization (including echo and status-update payloads that
must *not* produce a reply), idempotency under retries, bounded conversation
history, signature verification against tampered bodies, and the webhook
contract.

The agent tests are written against its failure modes rather than its happy
path: a hallucinated tool name, missing arguments, a tool that raises, a tool
that hangs until it is cancelled, a model that calls tools forever and never
answers, a model that answers with whitespace, and a double escalation that
must not open two handoffs.

The retrieval tests cover the same ground: an unrelated question returning
nothing, a synonym the lexical index provably cannot reach, and an internal
instruction that must never be forwarded to a customer — a bug this actually
had, kept as a regression test.

The LangGraph and MCP tests skip themselves when those optional packages are
absent, and CI runs a second job that installs everything at once — because
a dependency conflict between the optional runtimes and FastAPI is exactly
the failure a single green job would hide. The MCP tests speak the protocol
to the server through a real client session rather than calling its
handlers.

No network, no keys, no Redis — CI runs the same command.

## Configuration

| Variable | Effect when unset |
|---|---|
| `META_APP_SECRET` | signature verification is skipped |
| `META_VERIFY_TOKEN` | defaults to `dev-verify-token` |
| `META_GRAPH_TOKEN` | replies are logged instead of sent |
| `ANTHROPIC_API_KEY` | the scripted tool provider answers instead of a model |
| `VOYAGE_API_KEY` | retrieval is lexical (TF-IDF) instead of dense |
| `KNOWLEDGE_DIR` | defaults to `knowledge/` |
| `REDIS_URL` | state lives in process memory |

## Branches

`main` is what would be in production. `develop` is where changes land and
are tested.

Work happens on `develop` — directly for small changes, on a branch off it
for anything larger. CI runs on both, so nothing reaches `develop` without
the suite and the linter passing. Releasing is a pull request from
`develop` into `main`, which is protected: no direct pushes, no force
pushes, and both CI jobs have to be green before it can merge.

The point is that `main` is never a place where something is being tried
out. If it is on `main`, it survived `develop` first.

## Layout

```
app/
  main.py            FastAPI app: verification, webhook, background dispatch
  config.py          environment, read once
  channels/          per-channel parse/render adapters
  core/
    models.py        InboundMessage / OutboundMessage
    security.py      webhook signature verification
    store.py         conversation history + idempotency (memory or Redis)
    pipeline.py      what happens after the 200
    sender.py        logging sender / Graph API sender
  agent/
    tools.py         Tool Protocol, registry, built-in tools
    llm.py           tool-calling message types + scripted provider
    loop.py          the agent loop: budget, timeouts, error feedback
    langgraph_loop.py  the same agent as a LangGraph state machine
  mcp_server.py      the same tools over the Model Context Protocol
  rag/
    index.py         chunking, TF-IDF and dense indexes, the threshold
    embeddings.py    Embedder Protocol + Voyage embeddings
  ai/                provider Protocol, echo provider, Anthropic providers
knowledge/           the sample corpus, indexed at boot
tests/
```

## Notes

Written as a reference implementation of a pattern I have built in
production. The interesting parts are not the API calls — they are the
boundaries: one internal message model, one place where retries are
absorbed, one place where a failing provider stops being the user's problem.

MIT licensed.
