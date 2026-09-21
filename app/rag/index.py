"""Chunking, similarity search, and knowing when to say nothing.

The part of retrieval that gets written about is the search. The part
that decides whether the bot lies to a customer is the threshold: if the
best match is weak, this returns nothing at all, so the tool can tell the
model the knowledge base does not cover the question. A retriever that
always returns its top three results hands the model three irrelevant
paragraphs, and the model will do something with them.

Two indexes, one Protocol:

- `LexicalIndex` is the default. TF-IDF over the indexed corpus, no
  model, no key, no network. It matches on shared vocabulary, which is
  most support questions.
- `DenseIndex` takes an embedder and does the same thing with vectors
  from a model, for the questions lexical search cannot reach — "money
  back" against a document that says "refund".

Chunks carry their source so the reply can cite it. An answer nobody can
trace back to a document is not much better than a guess.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Big enough to hold a whole answer, small enough that a hit is
# specific. Support articles are short; this is not a legal corpus.
CHUNK_CHARS = 600

# Carried between chunks so an answer split across a boundary survives.
OVERLAP_CHARS = 120

# Below this, "no match" is the honest result. Calibrated against the
# sample corpus: a real hit lands around 0.25-0.6, an unrelated question
# below 0.05. It is a parameter and not a constant in the search call
# because a dense index scores on a different scale.
MIN_SCORE = 0.12

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that appear in nearly every chunk carry no signal but, unweighted,
# dominate the vector: "do you ship to Spain?" matched a paragraph about
# parcel tracking purely on "to" and "up". IDF already discounts them;
# removing them outright keeps short queries from being mostly noise.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can cannot do does for from had has
    have how i if in into is it its me my no not of on or our so than that
    the their them then there these they this to us was we were what when
    where which who will with would you your""".split()
)


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    ordinal: int
    """Position within its source, so a citation can point at a place."""

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.ordinal}"


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split on paragraphs, then pack up to `CHUNK_CHARS` with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(Chunk(buffer.strip(), source, len(chunks)))

    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) + 2 > CHUNK_CHARS:
            flush()
            buffer = buffer[-OVERLAP_CHARS:] if OVERLAP_CHARS else ""
        buffer = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph

    flush()
    return chunks


def tokenize(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
    # Bigrams give word order a little weight, so "not covered" and
    # "covered" do not land on exactly the same point.
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class VectorIndex(Protocol):
    async def add(self, chunks: list[Chunk]) -> None: ...
    async def search(
        self, query: str, limit: int = 3, min_score: float = MIN_SCORE
    ) -> list[tuple[Chunk, float]]: ...


class LexicalIndex:
    """TF-IDF over the indexed corpus. Sparse, exact, offline.

    IDF is what makes this work at all. Without it a query's common
    words outweigh the one term that carries its meaning, and the top
    hit is whichever chunk happens to be longest.

    IDF is recomputed on every `add`, which is fine for a corpus loaded
    once at boot and wrong for one that grows all day. That is the point
    at which this class should be replaced rather than patched.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._term_counts: list[Counter[str]] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunks.append(chunk)
            self._term_counts.append(Counter(tokenize(chunk.text)))
        self._reindex()

    def _reindex(self) -> None:
        total = len(self._term_counts)
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())

        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._vectors = [
            _normalize(self._weigh(counts)) for counts in self._term_counts
        ]

    def _weigh(self, counts: Counter[str]) -> dict[str, float]:
        # Sublinear term frequency: a word used five times is not five
        # times as relevant as a word used once.
        return {
            term: (1.0 + math.log(count)) * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }

    async def search(
        self, query: str, limit: int = 3, min_score: float = MIN_SCORE
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []

        query_vector = _normalize(self._weigh(Counter(tokenize(query))))
        if not query_vector:
            return []

        scored = [
            (chunk, _sparse_dot(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:limit] if pair[1] >= min_score]


class DenseIndex:
    """The same contract, backed by an embedding model.

    Exhaustive cosine over a list: linear in the number of chunks, which
    is the right trade for a few hundred support paragraphs and the
    wrong one past that. Swapping in pgvector or Qdrant does not touch
    the tool or the loop.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = await self._embedder.embed([chunk.text for chunk in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    async def search(
        self, query: str, limit: int = 3, min_score: float = MIN_SCORE
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []

        (query_vector,) = await self._embedder.embed([query])
        query_vector = _l2(query_vector)
        scored = [
            (chunk, sum(x * y for x, y in zip(query_vector, _l2(vector))))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [pair for pair in scored[:limit] if pair[1] >= min_score]


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def _sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    # Iterate the shorter side: a query has a handful of terms, a chunk
    # has hundreds.
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())


def _l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


async def load_directory(index: VectorIndex, directory: Path) -> int:
    """Index every markdown file in a directory. Returns chunks added."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name))
    await index.add(chunks)
    return len(chunks)
