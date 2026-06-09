"""Fournisseurs d'embeddings."""

__all__ = ["JinaEmbeddingProvider"]


def __getattr__(name: str):
    if name == "JinaEmbeddingProvider":
        from embeddings.jina_embeddings import JinaEmbeddingProvider

        return JinaEmbeddingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
