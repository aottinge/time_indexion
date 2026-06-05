#!/usr/bin/env python3
"""
Benchmark du pipeline RAG : mesure séparée de chaque étape.

Étapes mesurées :
  1. Extraction texte des PDF
  2. Conversion des pages PDF en images
  3. Chunking récursif
  4. Embedding texte
  5. Embedding image
  6. Indexation ChromaDB

Sorties : tableau terminal, CSV, Excel et graphique PNG.
"""

import argparse
import time
from pathlib import Path

import fitz

from chunkers.recursive_chunker import RecursiveChunker
from config import COLLECTION_BENCHMARK, PDF_DIR, REPORTS_DIR
from embeddings.jina_embeddings import JinaEmbeddingProvider
from reports.benchmark_report import (
    PipelineBenchmarkResult,
    generate_benchmark_chart,
    print_benchmark_table,
    results_to_dataframe,
    save_benchmark_csv,
    save_benchmark_excel,
    summary_to_dataframe,
)
from utils.pdf_document import PageRecord, PdfDocument, build_pdf_document
from utils.pdf_pages import extract_page_text_hybrid, render_page_image_fitz
from vectorstore.chroma_manager import ChromaManager


def _count_pages(pdf_files: list[Path]) -> int:
    total = 0
    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        try:
            total += doc.page_count
        finally:
            doc.close()
    return total


def run_pipeline_benchmark(
    pdf_dir: Path,
    label: str = "benchmark",
    use_ocr_fallback: bool = True,
) -> PipelineBenchmarkResult:
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF dans {pdf_dir}")

    num_pages = _count_pages(pdf_files)
    print(f"\n[1/7] Extraction texte ({len(pdf_files)} PDF, {num_pages} pages)...")
    page_records_by_pdf: dict[Path, list[PageRecord]] = {}

    t0 = time.perf_counter()
    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        try:
            n = doc.page_count
        finally:
            doc.close()

        records: list[PageRecord] = []
        for page_index in range(n):
            text, text_source = extract_page_text_hybrid(
                pdf_path,
                page_index,
                page_image=None,
                use_ocr_fallback=use_ocr_fallback,
            )
            records.append(
                PageRecord(
                    page_index=page_index,
                    text=text,
                    text_source=text_source,
                )
            )
        page_records_by_pdf[pdf_path] = records
    extraction_seconds = time.perf_counter() - t0

    print(f"[2/7] Conversion pages → images...")
    t0 = time.perf_counter()
    for pdf_path, records in page_records_by_pdf.items():
        for record in records:
            record.image = render_page_image_fitz(pdf_path, record.page_index)
            if not record.text.strip() and use_ocr_fallback and record.text_source == "empty":
                from utils.pdf_pages import extract_page_text_ocr

                ocr_text = extract_page_text_ocr(record.image)
                if ocr_text:
                    record.text = ocr_text
                    record.text_source = "ocr"
    image_render_seconds = time.perf_counter() - t0

    print("[3/7] Assemblage documents + chunking récursif...")
    documents: list[PdfDocument] = []
    for pdf_path, records in page_records_by_pdf.items():
        doc = build_pdf_document(pdf_path, page_records=records)
        if doc.content.strip():
            documents.append(doc)

    if not documents:
        raise FileNotFoundError("Aucun texte exploitable après extraction.")

    chunker = RecursiveChunker()
    t0 = time.perf_counter()
    chunk_result = chunker.chunk_documents(documents)
    chunking_seconds = time.perf_counter() - t0
    chunks = chunk_result.chunks
    chunks_with_images = sum(1 for c in chunks if c.images)

    print(f"      {len(chunks)} chunk(s) | {chunks_with_images} avec image(s)")

    print("[4/7] Chargement du modèle Jina v4...")
    t0 = time.perf_counter()
    provider = JinaEmbeddingProvider()
    model_load_seconds = time.perf_counter() - t0
    print(f"      Modèle chargé en {model_load_seconds:.2f}s")

    print("[5/7] Embedding texte...")
    text_embeddings, text_embedding_seconds = provider.embed_chunks_text_only_timed(chunks)

    print("[6/7] Embedding image...")
    image_embeddings, image_embedding_seconds, num_images_encoded = (
        provider.embed_chunks_images_only_timed(chunks)
    )

    print("[7/7] Fusion + indexation ChromaDB...")
    fused_embeddings = provider.fuse_hybrid_embeddings(
        text_embeddings, image_embeddings, chunks
    )
    chroma = ChromaManager()
    indexing_seconds = chroma.index_chunks(
        collection_name=COLLECTION_BENCHMARK,
        chunks=chunks,
        embeddings=fused_embeddings,
        embedding_mode="hybrid",
        reset=True,
    )

    return PipelineBenchmarkResult(
        label=label,
        extraction_seconds=extraction_seconds,
        image_render_seconds=image_render_seconds,
        chunking_seconds=chunking_seconds,
        text_embedding_seconds=text_embedding_seconds,
        image_embedding_seconds=image_embedding_seconds,
        indexing_seconds=indexing_seconds,
        model_load_seconds=model_load_seconds,
        num_documents=len(documents),
        num_pages=num_pages,
        num_chunks=len(chunks),
        chunks_with_images=chunks_with_images,
        num_images_encoded=num_images_encoded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pipeline RAG multimodal")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=PDF_DIR,
        help="Dossier contenant les PDF à indexer",
    )
    parser.add_argument(
        "--label",
        default="pipeline_recursive",
        help="Libellé du run (utilisé dans les exports)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Désactive le repli OCR pour les pages sans texte natif",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Benchmark pipeline RAG — temps par étape")
    print("=" * 60)

    result = run_pipeline_benchmark(
        pdf_dir=args.pdf_dir,
        label=args.label,
        use_ocr_fallback=not args.no_ocr,
    )

    steps_df = results_to_dataframe(result)
    summary_df = summary_to_dataframe(result)
    print_benchmark_table(steps_df, summary_df, result)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{args.label}"
    csv_path = REPORTS_DIR / f"{stem}.csv"
    excel_path = REPORTS_DIR / f"{stem}.xlsx"
    chart_path = REPORTS_DIR / f"{stem}.png"

    save_benchmark_csv(steps_df, summary_df, csv_path)
    save_benchmark_excel(steps_df, summary_df, excel_path)
    generate_benchmark_chart(result, chart_path)

    print(f"Rapports exportés :")
    print(f"  - {csv_path}")
    print(f"  - {csv_path.with_name(csv_path.stem + '_summary.csv')}")
    print(f"  - {excel_path}")
    print(f"  - {chart_path}")


if __name__ == "__main__":
    main()
