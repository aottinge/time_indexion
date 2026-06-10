"""
Extraction page par page : texte (natif ou OCR) + image (rendu PDF).
"""

import io
import shutil
from pathlib import Path

import fitz
from PIL import Image
from pypdf import PdfReader

from config import PAGE_RENDER_DPI


def _ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def render_page_image_fitz(pdf_path: Path, page_index: int, dpi: int = PAGE_RENDER_DPI) -> Image.Image:
    """Rend une page PyMuPDF en image PIL."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()


def extract_page_text_native(pdf_path: Path, page_index: int) -> str:
    """Texte natif d'une page via PyPDF."""
    reader = PdfReader(str(pdf_path))
    page = reader.pages[page_index]
    text = page.extract_text() or ""
    return text.strip()


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


def iter_pdf_pages(pdf_dir: Path, use_ocr_fallback: bool = True):
    """
    Parcourt chaque page de chaque PDF du dossier.

    Yields:
        dict avec source, page_index (0-based), text, text_source, image (PIL)
    """
    pdf_dir = Path(pdf_dir)
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(str(pdf_path))
        try:
            num_pages = doc.page_count
        finally:
            doc.close()

        for page_index in range(num_pages):
            image = render_page_image_fitz(pdf_path, page_index)
            text, text_source = extract_page_text_hybrid(
                pdf_path,
                page_index,
                page_image=image,
                use_ocr_fallback=use_ocr_fallback,
            )
            yield {
                "source": pdf_path.name,
                "page_index": page_index,
                "text": text,
                "text_source": text_source,
                "image": image,
            }
