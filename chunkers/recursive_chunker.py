"""
Chunking récursif via RecursiveCharacterTextSplitter (LangChain).
Chaque chunk est enrichi avec les images des pages PDF correspondantes.
"""

from chunkers.base import BaseChunker, Chunk
from chunkers.splitter_utils import create_recursive_splitter
from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from utils.pdf_document import PdfDocument, attach_images_to_chunks


class RecursiveChunker(BaseChunker):
    """Découpage récursif par tokens avec chevauchement + images de pages."""

    name = "recursive"

    def __init__(
        self,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        use_hf_tokenizer: bool | None = None,
    ) -> None:
        self._splitter = create_recursive_splitter(
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            use_hf_tokenizer=use_hf_tokenizer,
        )

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

            sub_texts = self._splitter.split_text(content)
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
