"""
Configuration centralisée du projet de comparaison de chunking RAG.
Modifier ces constantes pour adapter les expériences sans toucher au code métier.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "questions.json"
REPORTS_DIR = PROJECT_ROOT / "reports" / "output"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 256
SECTION_MAX_TOKENS = 1024
PAGE_RENDER_DPI = 150

JINA_MODEL = "jinaai/jina-embeddings-v4"
COLLECTION_RECURSIVE = "rag_recursive"
COLLECTION_SECTION = "rag_section"
COLLECTION_BENCHMARK = "benchmark_pipeline"

TOP_K = 5
