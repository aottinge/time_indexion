"""
Chargement des fichiers PDF depuis un dossier.
Utilise PyPDF, avec repli OCR (Tesseract) pour les PDF scannés sans couche texte.
"""

import io
import shutil
from pathlib import Path

from pypdf import PdfReader


def _extract_text_pypdf(pdf_path: Path) -> str:
    """Extraction texte standard (PDF avec couche texte)."""
    reader = PdfReader(str(pdf_path))
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def _extract_text_ocr(pdf_path: Path) -> str:
    """
    OCR page par page pour PDF scannés (images).
    Nécessite : pymupdf, pytesseract, pillow et le binaire `tesseract`.
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []

    print(f"      OCR en cours ({doc.page_count} pages) : {pdf_path.name}...")
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="fra+eng")
        if text.strip():
            pages_text.append(text.strip())

    return "\n\n".join(pages_text)


def _ocr_available() -> bool:
    return shutil.which("tesseract") is not None


def load_pdfs_from_directory(pdf_dir: Path, use_ocr_fallback: bool = True) -> list[dict[str, str]]:
    """
    Charge tous les PDF d'un dossier.

    Returns:
        Liste de dicts avec les clés 'source' (nom du fichier) et 'content' (texte).
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Dossier PDF introuvable : {pdf_dir}")

    documents: list[dict[str, str]] = []
    skipped: list[str] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans {pdf_dir}. "
            "Placez vos documents dans data/pdfs/."
        )

    for pdf_path in pdf_files:
        full_text = _extract_text_pypdf(pdf_path)

        if not full_text.strip() and use_ocr_fallback:
            if _ocr_available():
                try:
                    full_text = _extract_text_ocr(pdf_path)
                except ImportError as err:
                    skipped.append(
                        f"{pdf_path.name} (OCR indisponible: pip install pymupdf pytesseract)"
                    )
                    print(f"      Avertissement: {err}")
                    continue
            else:
                skipped.append(
                    f"{pdf_path.name} (PDF scanné sans texte — installez Tesseract: brew install tesseract)"
                )
                continue

        if full_text.strip():
            documents.append(
                {
                    "source": pdf_path.name,
                    "content": full_text,
                }
            )
        else:
            skipped.append(f"{pdf_path.name} (aucun texte extrait)")

    if not documents:
        details = "\n  - ".join(skipped) if skipped else "aucun fichier lisible"
        raise FileNotFoundError(
            f"Aucun contenu textuel extrait des PDF dans {pdf_dir}.\n"
            f"Fichiers concernés :\n  - {details}\n"
            "Si vos PDF sont des scans, installez Tesseract (`brew install tesseract`) "
            "et les dépendances OCR (`pip install pymupdf pytesseract`)."
        )

    if skipped:
        print(f"      Avertissement: {len(skipped)} fichier(s) ignoré(s):")
        for item in skipped:
            print(f"        - {item}")

    return documents
