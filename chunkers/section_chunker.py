"""
Chunking par section basé sur la détection de titres et sous-titres.
Chaque chunk est enrichi avec les images des pages PDF correspondantes.
"""

import re

from chunkers.base import BaseChunker, Chunk
from chunkers.text_splitter import (
    chars_for_tokens,
    count_tokens_approx,
    recursive_split_text,
)
from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    SECTION_MAX_TOKENS,
)
from utils.pdf_document import PdfDocument, attach_images_to_chunks

SECTION_PATTERNS = [
    r"^(?:Chapitre|Chapter|Partie|Part|Section)\s+[\dIVXLC]+[\.\:]?\s*.+$",
    r"^\d+(?:\.\d+)*\s+[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ][^\n]{2,120}$",
    r"^[IVXLC]+\.\s+[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ].{2,120}$",
    r"^[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ0-9\s\-]{4,80}$",
]

_COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_PATTERNS]


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 150:
        return False

    for pattern in _COMPILED_PATTERNS:
        if pattern.match(stripped):
            if pattern.pattern.endswith("{4,80}$"):
                if len(stripped.split()) > 12:
                    return False
            return True
    return False


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        if _is_section_header(line):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_title, content))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_title, content))

    return sections if sections else [("Document entier", text)]


class SectionChunker(BaseChunker):
    """Découpage par sections + images de pages."""

    name = "section"

    def __init__(
        self,
        max_section_tokens: int = SECTION_MAX_TOKENS,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._max_section_chars = chars_for_tokens(max_section_tokens)
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

            sections = _split_by_sections(content)
            doc_chunks: list[Chunk] = []

            for section_title, section_content in sections:
                section_tokens = count_tokens_approx(section_content)
                if section_tokens * 4 <= self._max_section_chars:
                    doc_chunks.append(
                        Chunk(
                            content=f"{section_title}\n\n{section_content}",
                            metadata={
                                "source": source,
                                "section_title": section_title,
                                "chunk_index": global_index,
                                "strategy": self.name,
                            },
                        )
                    )
                    global_index += 1
                else:
                    sub_texts = recursive_split_text(
                        section_content, self._chunk_size, self._chunk_overlap
                    )
                    for j, sub_text in enumerate(sub_texts):
                        doc_chunks.append(
                            Chunk(
                                content=f"{section_title}\n\n{sub_text}",
                                metadata={
                                    "source": source,
                                    "section_title": section_title,
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
