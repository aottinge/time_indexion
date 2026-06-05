"""
Provider d'embeddings Jina v4 en local (Sentence Transformers).
Support texte seul, images seules et fusion hybride (texte + images).
"""

import math
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from chunkers.base import Chunk
from config import JINA_MODEL


def _to_numpy(out: Any) -> np.ndarray:
    """Convertit la sortie (ndarray, tensor, liste de tensors) en numpy CPU."""
    import torch

    if isinstance(out, np.ndarray):
        return out.astype(np.float32)

    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy().astype(np.float32)

    if isinstance(out, (list, tuple)):
        if not out:
            return np.empty((0,))
        if all(isinstance(x, torch.Tensor) for x in out):
            return torch.stack([x.detach().cpu() for x in out]).numpy().astype(np.float32)
        return np.stack([_to_numpy(x) for x in out])

    return np.asarray(out, dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class JinaEmbeddingProvider:
    """
    Client local Jina Embeddings v4 via SentenceTransformer.
    - texte : model.encode()
    - images : encode_image() du modèle Jina sous-jacent
    """

    def __init__(
        self,
        model_name: str = JINA_MODEL,
        device: str | None = None,
        batch_size: int = 4,
    ) -> None:
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        self._model.eval()

    def _resolve_jina_core(self) -> Any:
        """Accède au modèle Jina multimodal (encode_text / encode_image) sous-jacent."""
        candidates: list[Any] = [self._model]

        if hasattr(self._model, "_first_module"):
            try:
                candidates.append(self._model._first_module())
            except Exception:
                pass

        try:
            for idx in range(len(self._model)):
                candidates.append(self._model[idx])
        except Exception:
            pass

        for mod in candidates:
            if mod is None:
                continue
            core = getattr(mod, "auto_model", mod)
            if mod is not core:
                candidates.append(core)
            if hasattr(core, "encode_text") and hasattr(core, "encode_image"):
                return core

        raise RuntimeError(
            "Modèle Jina multimodal indisponible via SentenceTransformer. "
            "Vérifiez sentence-transformers et jinaai/jina-embeddings-v4."
        )

    def _encode_texts(self, texts: list[str], prompt_name: str = "passage") -> np.ndarray:
        batches = math.ceil(len(texts) / self.batch_size) if texts else 0
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            print(
                f"      Embeddings batch texte "
                f"({i // self.batch_size + 1}/{max(batches, 1)})..."
            )
            out = self._model.encode(
                sentences=batch,
                task="retrieval",
                prompt_name=prompt_name,
                batch_size=len(batch),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            arr = _to_numpy(out)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                all_vecs.append(_normalize(row))

        return np.stack(all_vecs) if all_vecs else np.empty((0,))

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0,))

        core = self._resolve_jina_core()
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            print(
                f"      Embeddings batch image "
                f"({i // self.batch_size + 1}/{math.ceil(len(images) / self.batch_size)})..."
            )
            out = core.encode_image(images=batch, task="retrieval", return_numpy=True)
            arr = _to_numpy(out)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for row in arr:
                all_vecs.append(_normalize(row))

        return np.stack(all_vecs)

    @staticmethod
    def _unique_page_map(chunks: list[Chunk]) -> dict[tuple[str, int], Image.Image]:
        """Regroupe les images de pages uniques (source, page_index)."""
        page_map: dict[tuple[str, int], Image.Image] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            page_indices = chunk.metadata.get("page_indices", [])
            for page_idx, image in zip(page_indices, chunk.images):
                key = (source, page_idx)
                if key not in page_map:
                    page_map[key] = image
        return page_map

    def encode_pages_once(
        self, chunks: list[Chunk]
    ) -> tuple[dict[tuple[str, int], np.ndarray], int]:
        """
        Encode chaque page PDF une seule fois.
        Retourne (cache (source, page_index) → vecteur, nombre de pages encodées).
        """
        page_map = self._unique_page_map(chunks)
        if not page_map:
            return {}, 0

        keys = sorted(page_map.keys())
        images = [page_map[key] for key in keys]
        print(f"      {len(images)} page(s) unique(s) à encoder...")
        vecs = self._encode_images(images)
        cache = {key: _normalize(vecs[i]) for i, key in enumerate(keys)}
        return cache, len(keys)

    def _chunk_image_vec_from_cache(
        self, chunk: Chunk, page_cache: dict[tuple[str, int], np.ndarray]
    ) -> np.ndarray | None:
        source = chunk.metadata.get("source", "")
        page_indices = chunk.metadata.get("page_indices", [])
        if not page_indices:
            return None

        vecs = [
            page_cache[(source, page_idx)]
            for page_idx in page_indices
            if (source, page_idx) in page_cache
        ]
        if not vecs:
            return None
        return _normalize(np.mean(np.stack(vecs), axis=0))

    def _embed_single_multimodal(
        self,
        chunk: Chunk,
        page_cache: dict[tuple[str, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        text = chunk.content.strip() or "[contenu visuel]"
        text_vec = _normalize(
            _to_numpy(
                self._model.encode(
                    sentences=[text],
                    task="retrieval",
                    prompt_name="passage",
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            ).reshape(-1)
        )

        if not chunk.images:
            return text_vec

        if page_cache is not None:
            img_vec = self._chunk_image_vec_from_cache(chunk, page_cache)
            if img_vec is None:
                return text_vec
            fused = np.mean(np.vstack([text_vec, img_vec]), axis=0)
            return _normalize(fused)

        img_emb = self._encode_images(chunk.images)
        fused = np.mean(np.vstack([text_vec, *img_emb]), axis=0)
        return _normalize(fused)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._encode_texts(texts)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = self._encode_texts([text], prompt_name="query")[0]
        return vec.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        page_cache, _ = self.encode_pages_once(chunks)
        embeddings: list[list[float]] = []
        for i, chunk in enumerate(chunks, start=1):
            n_img = len(chunk.metadata.get("page_indices", []))
            print(f"      Embedding multimodal {i}/{len(chunks)} ({n_img} page(s))...")
            vec = self._embed_single_multimodal(chunk, page_cache=page_cache)
            embeddings.append(vec.tolist())
        return embeddings

    def embed_chunks_text_only(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [c.content for c in chunks]
        vecs = self._encode_texts(texts)
        return [v.tolist() for v in vecs]

    def embed_chunks_images_only(self, chunks: list[Chunk]) -> tuple[list[list[float]], int]:
        page_cache, num_pages_encoded = self.encode_pages_once(chunks)
        embeddings: list[list[float]] = []

        for chunk in chunks:
            vec = self._chunk_image_vec_from_cache(chunk, page_cache)
            if vec is None:
                embeddings.append([0.0])
            else:
                embeddings.append(vec.tolist())

        return embeddings, num_pages_encoded

    def fuse_hybrid_embeddings(
        self,
        text_embeddings: list[list[float]],
        image_embeddings: list[list[float]],
        chunks: list[Chunk],
    ) -> list[list[float]]:
        fused: list[list[float]] = []
        for text_emb, img_emb, chunk in zip(text_embeddings, image_embeddings, chunks):
            t = _normalize(np.asarray(text_emb, dtype=np.float32))
            if chunk.images and img_emb and img_emb != [0.0]:
                im = _normalize(np.asarray(img_emb, dtype=np.float32))
                vec = _normalize((t + im) / 2.0)
            else:
                vec = t
            fused.append(vec.tolist())
        return fused

    def embed_documents_timed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        start = time.perf_counter()
        embeddings = self.embed_documents(texts)
        return embeddings, time.perf_counter() - start

    def embed_chunks_timed(self, chunks: list[Chunk]) -> tuple[list[list[float]], float]:
        start = time.perf_counter()
        embeddings = self.embed_chunks(chunks)
        return embeddings, time.perf_counter() - start

    def embed_chunks_text_only_timed(
        self, chunks: list[Chunk]
    ) -> tuple[list[list[float]], float]:
        start = time.perf_counter()
        embeddings = self.embed_chunks_text_only(chunks)
        return embeddings, time.perf_counter() - start

    def embed_chunks_images_only_timed(
        self, chunks: list[Chunk]
    ) -> tuple[list[list[float]], float, int]:
        start = time.perf_counter()
        embeddings, n_images = self.embed_chunks_images_only(chunks)
        return embeddings, time.perf_counter() - start, n_images
