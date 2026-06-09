#!/usr/bin/env python3
"""
Benchmark du pipeline RAG : mesure séparée de chaque étape.

Étapes mesurées :
  1. Extraction texte des PDF (+ rendu images si mode hybride)
  2. Chunking récursif
  3. Embedding texte (optionnel)
  4. Embedding image (optionnel)
  5. Indexation ChromaDB (optionnel)

Sorties : tableau terminal, CSV, Excel et graphique PNG.
"""

import argparse
import time
from pathlib import Path

from config import COLLECTION_BENCHMARK, PDF_DIR, REPORTS_DIR


def _count_pages(pdf_files: list[Path]) -> int:
    import fitz

    total = 0
    for pdf_path in pdf_files:
        doc = fitz.open(str(pdf_path))
        try:
            total += doc.page_count
        finally:
            doc.close()
    return total


def _extract_pdfs(
    pdf_files: list[Path],
    *,
    use_ocr_fallback: bool,
    text_only: bool,
) -> tuple[dict[Path, list], float]:
    from utils.pdf_document import PageRecord
    from utils.pdf_pages import extract_pages_from_pdf

    page_records_by_pdf: dict[Path, list[PageRecord]] = {}
    t0 = time.perf_counter()

    for pdf_path in pdf_files:
        records: list[PageRecord] = []
        for page_data in extract_pages_from_pdf(
            pdf_path,
            use_ocr_fallback=use_ocr_fallback,
            render_images=not text_only,
        ):
            records.append(
                PageRecord(
                    page_index=page_data["page_index"],
                    text=page_data["text"],
                    text_source=page_data["text_source"],
                    image=page_data["image"],
                )
            )
        page_records_by_pdf[pdf_path] = records

    return page_records_by_pdf, time.perf_counter() - t0


def run_pipeline_benchmark(
    pdf_dir: Path,
    label: str = "benchmark",
    use_ocr_fallback: bool = True,
    text_only: bool = False,
    skip_embeddings: bool = False,
    skip_index: bool = False,
    use_hf_tokenizer: bool = False,
) -> "PipelineBenchmarkResult":
    from chunkers.recursive_chunker import RecursiveChunker
    from reports.benchmark_report import PipelineBenchmarkResult
    from utils.pdf_document import PdfDocument, build_pdf_document

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF dans {pdf_dir}")

    num_pages = _count_pages(pdf_files)
    mode_label = "texte seul" if text_only else "hybride"
    print(f"\n[1/6] Extraction PDF ({mode_label}) — {len(pdf_files)} PDF, {num_pages} pages...")

    page_records_by_pdf, extraction_seconds = _extract_pdfs(
        pdf_files,
        use_ocr_fallback=use_ocr_fallback,
        text_only=text_only,
    )
    image_render_seconds = 0.0

    print("[2/6] Assemblage documents + chunking récursif...")
    documents: list[PdfDocument] = []
    for pdf_path, records in page_records_by_pdf.items():
        doc = build_pdf_document(pdf_path, page_records=records)
        if doc.content.strip():
            documents.append(doc)

    if not documents:
        raise FileNotFoundError("Aucun texte exploitable après extraction.")

    chunker = RecursiveChunker(use_hf_tokenizer=use_hf_tokenizer)
    t0 = time.perf_counter()
    chunk_result = chunker.chunk_documents(documents)
    chunking_seconds = time.perf_counter() - t0
    chunks = chunk_result.chunks
    chunks_with_images = sum(1 for c in chunks if c.images)

    print(f"      {len(chunks)} chunk(s) | {chunks_with_images} avec image(s)")

    if skip_embeddings:
        return PipelineBenchmarkResult(
            label=label,
            extraction_seconds=extraction_seconds,
            image_render_seconds=image_render_seconds,
            chunking_seconds=chunking_seconds,
            text_embedding_seconds=0.0,
            image_embedding_seconds=0.0,
            indexing_seconds=0.0,
            model_load_seconds=0.0,
            num_documents=len(documents),
            num_pages=num_pages,
            num_chunks=len(chunks),
            chunks_with_images=chunks_with_images,
            num_images_encoded=0,
        )

    from embeddings.jina_embeddings import JinaEmbeddingProvider

    print("[3/6] Chargement du modèle Jina v4...")
    t0 = time.perf_counter()
    provider = JinaEmbeddingProvider()
    model_load_seconds = time.perf_counter() - t0
    print(f"      Modèle chargé en {model_load_seconds:.2f}s")

    print("[4/6] Embedding texte...")
    text_embeddings, text_embedding_seconds = provider.embed_chunks_text_only_timed(chunks)

    image_embedding_seconds = 0.0
    num_images_encoded = 0
    if text_only:
        image_embeddings = [[0.0] for _ in chunks]
    else:
        print("[5/6] Embedding image...")
        image_embeddings, image_embedding_seconds, num_images_encoded = (
            provider.embed_chunks_images_only_timed(chunks)
        )

    indexing_seconds = 0.0
    fused_embeddings = provider.fuse_hybrid_embeddings(
        text_embeddings, image_embeddings, chunks
    )

    if not skip_index:
        print("[6/6] Indexation ChromaDB...")
        from vectorstore.chroma_manager import ChromaManager

        chroma = ChromaManager()
        indexing_seconds = chroma.index_chunks(
            collection_name=COLLECTION_BENCHMARK,
            chunks=chunks,
            embeddings=fused_embeddings,
            embedding_mode="text" if text_only else "hybrid",
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
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Mode texte seul : pas d'images ni d'embeddings visuels (plus léger)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="S'arrête après le chunking (évite torch/Jina)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Calcule les embeddings mais n'indexe pas dans ChromaDB",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="N'exporte pas CSV, Excel ni graphique PNG",
    )
    parser.add_argument(
        "--hf-tokenizer",
        action="store_true",
        help="Utilise le tokenizer HuggingFace Jina (plus lent au démarrage)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Benchmark pipeline RAG — temps par étape")
    print("=" * 60)

    result = run_pipeline_benchmark(
        pdf_dir=args.pdf_dir,
        label=args.label,
        use_ocr_fallback=not args.no_ocr,
        text_only=args.text_only,
        skip_embeddings=args.skip_embeddings,
        skip_index=args.skip_index,
        use_hf_tokenizer=args.hf_tokenizer,
    )

    from reports.benchmark_report import (
        generate_benchmark_chart,
        print_benchmark_table,
        results_to_dataframe,
        save_benchmark_csv,
        save_benchmark_excel,
        summary_to_dataframe,
    )

    steps_df = results_to_dataframe(result)
    summary_df = summary_to_dataframe(result)
    print_benchmark_table(steps_df, summary_df, result)

    if args.no_export:
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{args.label}"
    csv_path = REPORTS_DIR / f"{stem}.csv"
    excel_path = REPORTS_DIR / f"{stem}.xlsx"
    chart_path = REPORTS_DIR / f"{stem}.png"

    save_benchmark_csv(steps_df, summary_df, csv_path)
    save_benchmark_excel(steps_df, summary_df, excel_path)
    generate_benchmark_chart(result, chart_path)

    print("Rapports exportés :")
    print(f"  - {csv_path}")
    print(f"  - {csv_path.with_name(csv_path.stem + '_summary.csv')}")
    print(f"  - {excel_path}")
    print(f"  - {chart_path}")


if __name__ == "__main__":
    main()
