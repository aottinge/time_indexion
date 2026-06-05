"""
Rapport du benchmark pipeline : tableau terminal, CSV, Excel et graphique.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PipelineBenchmarkResult:
    """Métriques mesurées sur un jeu de PDF."""

    label: str
    extraction_seconds: float
    image_render_seconds: float
    chunking_seconds: float
    text_embedding_seconds: float
    image_embedding_seconds: float
    indexing_seconds: float
    model_load_seconds: float
    num_documents: int
    num_pages: int
    num_chunks: int
    chunks_with_images: int
    num_images_encoded: int

    @property
    def measured_total_seconds(self) -> float:
        return (
            self.extraction_seconds
            + self.image_render_seconds
            + self.chunking_seconds
            + self.text_embedding_seconds
            + self.image_embedding_seconds
            + self.indexing_seconds
        )


def results_to_dataframe(result: PipelineBenchmarkResult) -> pd.DataFrame:
    """Tableau des étapes avec temps absolus et part du total mesuré."""
    total = result.measured_total_seconds or 1.0

    rows = [
        {
            "Étape": "Extraction PDF (texte)",
            "Temps (s)": round(result.extraction_seconds, 4),
            "% du pipeline": round(100 * result.extraction_seconds / total, 1),
            "Détail": f"{result.num_documents} doc(s), {result.num_pages} page(s)",
        },
        {
            "Étape": "Conversion pages → images",
            "Temps (s)": round(result.image_render_seconds, 4),
            "% du pipeline": round(100 * result.image_render_seconds / total, 1),
            "Détail": f"{result.num_pages} page(s) rendues",
        },
        {
            "Étape": "Chunking récursif",
            "Temps (s)": round(result.chunking_seconds, 4),
            "% du pipeline": round(100 * result.chunking_seconds / total, 1),
            "Détail": f"{result.num_chunks} chunk(s)",
        },
        {
            "Étape": "Embedding texte",
            "Temps (s)": round(result.text_embedding_seconds, 4),
            "% du pipeline": round(100 * result.text_embedding_seconds / total, 1),
            "Détail": f"{result.num_chunks} encodage(s) texte",
        },
        {
            "Étape": "Embedding image",
            "Temps (s)": round(result.image_embedding_seconds, 4),
            "% du pipeline": round(100 * result.image_embedding_seconds / total, 1),
            "Détail": f"{result.num_images_encoded} page(s) unique(s) encodée(s)",
        },
        {
            "Étape": "Indexation ChromaDB",
            "Temps (s)": round(result.indexing_seconds, 4),
            "% du pipeline": round(100 * result.indexing_seconds / total, 1),
            "Détail": f"{result.num_chunks} vecteur(s)",
        },
        {
            "Étape": "TOTAL (6 étapes)",
            "Temps (s)": round(result.measured_total_seconds, 4),
            "% du pipeline": 100.0,
            "Détail": f"{result.chunks_with_images} chunk(s) avec image(s)",
        },
    ]
    return pd.DataFrame(rows)


def summary_to_dataframe(result: PipelineBenchmarkResult) -> pd.DataFrame:
    """Données complémentaires pour l'export."""
    rows = [
        {"Indicateur": "Documents", "Valeur": result.num_documents},
        {"Indicateur": "Pages", "Valeur": result.num_pages},
        {"Indicateur": "Chunks", "Valeur": result.num_chunks},
        {"Indicateur": "Chunks avec images", "Valeur": result.chunks_with_images},
        {
            "Indicateur": "Chargement modèle (hors graphique)",
            "Valeur": round(result.model_load_seconds, 4),
        },
        {"Indicateur": "Libellé", "Valeur": result.label},
    ]
    return pd.DataFrame(rows)


def print_benchmark_table(
    steps_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    result: PipelineBenchmarkResult,
) -> None:
    print("\n" + "=" * 80)
    print("  BENCHMARK PIPELINE — TEMPS PAR ÉTAPE")
    print("=" * 80 + "\n")
    print(steps_df.to_string(index=False))
    print(f"\n  Chargement modèle Jina (non inclus dans le graphique) : "
          f"{result.model_load_seconds:.4f}s")
    print("\n--- Résumé ---\n")
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 80 + "\n")


def save_benchmark_csv(steps_df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_df.to_csv(path, index=False, encoding="utf-8-sig")
    summary_path = path.with_name(f"{path.stem}_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def save_benchmark_excel(steps_df: pd.DataFrame, summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        steps_df.to_excel(writer, index=False, sheet_name="Temps par étape")
        summary_df.to_excel(writer, index=False, sheet_name="Résumé")


def generate_benchmark_chart(result: PipelineBenchmarkResult, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [
        "Extraction\ntexte",
        "Pages →\nimages",
        "Chunking\nrécursif",
        "Embedding\ntexte",
        "Embedding\nimage",
        "Indexation\nChromaDB",
    ]
    values = [
        result.extraction_seconds,
        result.image_render_seconds,
        result.chunking_seconds,
        result.text_embedding_seconds,
        result.image_embedding_seconds,
        result.indexing_seconds,
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Temps (secondes)")
    ax.set_title(
        f"Benchmark pipeline — {result.num_chunks} chunks | "
        f"{result.num_pages} pages | {result.num_images_encoded} images",
        fontsize=12,
        fontweight="bold",
    )

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
