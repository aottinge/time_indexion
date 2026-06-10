"""
Indexation et recherche ChromaDB.
"""

import time
import uuid

import chromadb
from chromadb.config import Settings

from chunkers.base import Chunk
from config import CHROMA_PERSIST_DIR, TOP_K


class ChromaManager:
    def __init__(self, persist_directory: str | None = None) -> None:
        path = persist_directory or str(CHROMA_PERSIST_DIR)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedding_mode: str = "hybrid",
        reset: bool = True,
    ) -> float:
        """
        Indexe les chunks dans ChromaDB.
        Retourne le temps d'indexation en secondes.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Nombre de chunks ({len(chunks)}) != embeddings ({len(embeddings)})"
            )

        if reset:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass

        collection = self.get_or_create_collection(collection_name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(str(uuid.uuid4()))
            documents.append(chunk.content)
            meta = dict(chunk.metadata)
            meta["embedding_mode"] = embedding_mode
            metadatas.append(meta)

        start = time.perf_counter()
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return time.perf_counter() - start

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = TOP_K,
    ) -> dict:
        collection = self._client.get_collection(collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )
