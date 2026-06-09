"""
Construction de documents PDF avec pages (texte + image) et mapping char → page.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from utils.pdf_pages import extract_pages_from_pdf


@dataclass
class PageRecord:
    page_index: int
    text: str
    text_source: str
    image: Optional[Image.Image] = None


@dataclass
class PdfDocument:
    source: str
    content: str
    pdf_path: Path
    pages: list[PageRecord] = field(default_factory=list)
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)


def build_pdf_document(
    pdf_path: Path,
    page_records: list[PageRecord] | None = None,
    use_ocr_fallback: bool = True,
) -> PdfDocument:
    """
    Assemble le texte complet et les métadonnées page par page.
    Si page_records est fourni, évite une nouvelle extraction/rendu.
    """
    pdf_path = Path(pdf_path)
    source = pdf_path.name

    if page_records is None:
        page_records = []
        for page_data in extract_pages_from_pdf(
            pdf_path,
            use_ocr_fallback=use_ocr_fallback,
            render_images=True,
        ):
            page_records.append(
                PageRecord(
                    page_index=page_data["page_index"],
                    text=page_data["text"],
                    text_source=page_data["text_source"],
                    image=page_data["image"],
                )
            )

    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0

    for record in page_records:
        block = record.text.strip()
        if not block:
            continue
        if parts:
            separator = "\n\n"
            offset += len(separator)
            parts.append(separator)
        start = offset
        parts.append(block)
        offset += len(block)
        spans.append((start, offset, record.page_index))

    full_text = "".join(parts)
    return PdfDocument(
        source=source,
        content=full_text,
        pdf_path=pdf_path,
        pages=page_records,
        page_spans=spans,
    )


def load_documents_with_pages(pdf_dir: Path, use_ocr_fallback: bool = True) -> list[PdfDocument]:
    """Charge tous les PDF avec pages texte+image."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Dossier PDF introuvable : {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun fichier PDF dans {pdf_dir}. Placez vos documents dans data/pdfs/."
        )

    documents: list[PdfDocument] = []
    for pdf_path in pdf_files:
        doc = build_pdf_document(pdf_path, use_ocr_fallback=use_ocr_fallback)
        if doc.content.strip():
            documents.append(doc)

    if not documents:
        raise FileNotFoundError(f"Aucun contenu exploitable dans {pdf_dir}.")

    return documents


def page_indices_for_span(pdf_doc: PdfDocument, start: int, end: int) -> list[int]:
    """Retourne les indices de pages chevauchant l'intervalle [start, end)."""
    indices: list[int] = []
    for span_start, span_end, page_idx in pdf_doc.page_spans:
        if span_end > start and span_start < end:
            if page_idx not in indices:
                indices.append(page_idx)
    return indices


def attach_images_to_chunks(chunks: list, pdf_doc: PdfDocument, full_text: str) -> None:
    """
    Associe à chaque chunk les images des pages correspondantes.
    Modifie les chunks en place (attributs image, images, metadata).
    """
    search_from = 0
    for chunk in chunks:
        needle = chunk.content[:80].strip()
        if not needle:
            continue

        pos = full_text.find(needle, search_from)
        if pos < 0:
            pos = full_text.find(needle)
        if pos < 0:
            continue

        search_from = pos
        end = pos + len(chunk.content)
        page_idxs = page_indices_for_span(pdf_doc, pos, end)

        images: list[Image.Image] = []
        for page_idx in page_idxs:
            for record in pdf_doc.pages:
                if record.page_index == page_idx:
                    images.append(record.image)
                    break

        chunk.images = images
        chunk.image = images[0] if images else None
        chunk.metadata["page_indices"] = page_idxs
        chunk.metadata["text_source"] = "text+image" if images else "text"
        chunk.metadata["modality"] = "hybrid" if images else "text"
