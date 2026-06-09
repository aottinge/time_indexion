"""
Extraction page par page : texte (natif ou OCR) + image (rendu PDF).
"""

import io
import shutil
from pathlib import Path
from typing import TypedDict

import fitz
from PIL import Image

from config import PAGE_RENDER_DPI


class PageData(TypedDict):
    page_index: int
    text: str
    text_source: str
    image: Image.Image | None


def _ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def _render_page_image(page: fitz.Page, dpi: int = PAGE_RENDER_DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def render_page_image_fitz(pdf_path: Path, page_index: int, dpi: int = PAGE_RENDER_DPI) -> Image.Image:
    """Rend une page PyMuPDF en image PIL."""
    doc = fitz.open(str(pdf_path))
    try:
        return _render_page_image(doc[page_index], dpi=dpi)
    finally:
        doc.close()


def extract_page_text_native(pdf_path: Path, page_index: int) -> str:
    """Texte natif d'une page via PyMuPDF (une seule ouverture du PDF)."""
    doc = fitz.open(str(pdf_path))
    try:
        return (doc[page_index].get_text() or "").strip()
    finally:
        doc.close()


def extract_page_text_ocr(page_image: Image.Image) -> str:
    """OCR sur l'image de page."""
    import pytesseract

    return pytesseract.image_to_string(page_image, lang="fra+eng").strip()


def extract_page_text_hybrid(
    pdf_path: Path,
    page_index: int,
    page_image: Image.Image | None = None,
    use_ocr_fallback: bool = True,
) -> tuple[str, str]:
    """
    Texte d'une page : extraction native, puis OCR si vide.

    Returns:
        (texte, source) avec source in ('native', 'ocr', 'empty')
    """
    text = extract_page_text_native(pdf_path, page_index)
    if text:
        return text, "native"

    if use_ocr_fallback and _ocr_available():
        if page_image is None:
            page_image = render_page_image_fitz(pdf_path, page_index)
        ocr_text = extract_page_text_ocr(page_image)
        if ocr_text:
            return ocr_text, "ocr"

    return "", "empty"


def extract_pages_from_pdf(
    pdf_path: Path,
    *,
    use_ocr_fallback: bool = True,
    render_images: bool = True,
    dpi: int = PAGE_RENDER_DPI,
) -> list[PageData]:
    """
    Parcourt toutes les pages d'un PDF en une seule ouverture de fichier.
    """
    pdf_path = Path(pdf_path)
    pages: list[PageData] = []

    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            text = (page.get_text() or "").strip()
            text_source = "native" if text else "empty"
            image: Image.Image | None = None

            needs_image = render_images or (
                use_ocr_fallback and not text and _ocr_available()
            )
            if needs_image:
                image = _render_page_image(page, dpi=dpi)

            if not text and use_ocr_fallback and image is not None:
                ocr_text = extract_page_text_ocr(image)
                if ocr_text:
                    text = ocr_text
                    text_source = "ocr"

            pages.append(
                {
                    "page_index": page_index,
                    "text": text,
                    "text_source": text_source,
                    "image": image if render_images else None,
                }
            )
    finally:
        doc.close()

    return pages


def iter_pdf_pages(pdf_dir: Path, use_ocr_fallback: bool = True):
    """
    Parcourt chaque page de chaque PDF du dossier.

    Yields:
        dict avec source, page_index (0-based), text, text_source, image (PIL)
    """
    pdf_dir = Path(pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        for page_data in extract_pages_from_pdf(
            pdf_path,
            use_ocr_fallback=use_ocr_fallback,
            render_images=True,
        ):
            yield {
                "source": pdf_path.name,
                "page_index": page_data["page_index"],
                "text": page_data["text"],
                "text_source": page_data["text_source"],
                "image": page_data["image"],
            }
