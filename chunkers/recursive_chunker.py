"""
Chunking récursif par caractères.
Chaque chunk est enrichi avec les images des pages PDF correspondantes.
"""

from chunkers.base import BaseChunker, Chunk
from chunkers.text_splitter import chars_for_tokens, recursive_split_text
from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from utils.pdf_document import PdfDocument, attach_images_to_chunks


class RecursiveChunker(BaseChunker):
    """Découpage récursif avec chevauchement + images de pages."""

    name = "recursive"

    def __init__(
        self,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._chunk_size = chars_for_tokens(chunk_size_tokens)
        self._chunk_overlap = chars_for_tokens(chunk_overlap_tokens)

    def _split(self, documents: list) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        global_index = 0

        for doc in documents:
            if isinstance(doc, PdfDocument):
                pdf_doc = doc
                source = pdf_doc.source
                content = pdf_doc.content
            else:
                source = doc["source"]
                content = doc["content"]
                pdf_doc = None

            sub_texts = recursive_split_text(
                content, self._chunk_size, self._chunk_overlap
            )
            doc_chunks: list[Chunk] = []

            for j, sub_text in enumerate(sub_texts):
                doc_chunks.append(
                    Chunk(
                        content=sub_text,
                        metadata={
                            "source": source,
                            "sub_chunk": j,
                            "chunk_index": global_index,
                            "strategy": self.name,
                        },
                    )
                )
                global_index += 1

            if pdf_doc is not None:
                attach_images_to_chunks(doc_chunks, pdf_doc, content)

            all_chunks.extend(doc_chunks)

        return all_chunks
