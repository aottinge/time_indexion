"""
Interface de base pour les stratégies de chunking.
Permet d'ajouter facilement de nouvelles stratégies en héritant de BaseChunker.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Fragment de texte avec métadonnées et images de pages PDF associées."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image: Any = None
    images: list[Any] = field(default_factory=list)


@dataclass
class ChunkResult:
    """Résultat d'une opération de chunking."""

    chunks: list[Chunk]
    chunking_time_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def average_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(c.content) for c in self.chunks) / len(self.chunks)


class BaseChunker(ABC):
    """Contrat commun pour toutes les stratégies de découpage."""

    name: str = "base"

    def chunk_documents(self, documents: list) -> ChunkResult:
        """
        Découpe une liste de documents et mesure le temps écoulé.

        Args:
            documents: Liste de dicts ou PdfDocument.
        """
        start = time.perf_counter()
        chunks = self._split(documents)
        elapsed = time.perf_counter() - start
        return ChunkResult(chunks=chunks, chunking_time_seconds=elapsed)

    @abstractmethod
    def _split(self, documents: list) -> list[Chunk]:
        """Implémentation spécifique de la stratégie."""
        ...
