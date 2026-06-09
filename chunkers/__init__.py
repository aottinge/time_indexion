"""Stratégies de découpage de documents."""

from chunkers.base import BaseChunker, Chunk, ChunkResult

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkResult",
    "RecursiveChunker",
    "SectionChunker",
]


def __getattr__(name: str):
    if name == "RecursiveChunker":
        from chunkers.recursive_chunker import RecursiveChunker

        return RecursiveChunker
    if name == "SectionChunker":
        from chunkers.section_chunker import SectionChunker

        return SectionChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
