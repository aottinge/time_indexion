"""
Génération des rapports comparatifs : terminal, CSV, Excel et graphiques.
"""

from pathlib import Path

import pandas as pd

from evaluation.evaluator import StrategyMetrics


def metrics_to_dataframe(metrics_list: list[StrategyMetrics]) -> pd.DataFrame:
    """Convertit les métriques en DataFrame pour export et affichage."""
    rows = []
    for m in metrics_list:
        rows.append(
            {
                "Stratégie": m.strategy_name,
                "Nombre total de chunks": m.total_chunks,
                "Taille moyenne des chunks (car.)": round(m.average_chunk_size, 1),
                "Temps chunking (s)": round(m.chunking_time_seconds, 4),
                "Temps embedding (s)": round(m.embedding_time_seconds, 4),
                "Temps indexation Chroma (s)": round(m.indexing_time_seconds, 4),
                "Temps indexation total (s)": round(m.total_indexation_time_seconds, 4),
                "Temps moyen recherche (s)": round(m.average_search_time_seconds, 4),
                "Questions testées": m.total_questions,
                "Score moyen similarité": round(m.average_similarity_score, 4),
                "Score max similarité": round(m.max_similarity_score, 4),
            }
        )
    return pd.DataFrame(rows)


def print_comparison_table(df: pd.DataFrame) -> None:
    """Affiche un tableau comparatif dans le terminal."""
    print("\n" + "=" * 80)
    print("  RAPPORT COMPARATIF — STRATÉGIES DE CHUNKING RAG")
    print("=" * 80 + "\n")
    print(df.to_string(index=False))
    print("\n" + "=" * 80 + "\n")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Exporte le rapport en CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_excel(df: pd.DataFrame, path: Path) -> None:
    """Exporte le rapport en Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, sheet_name="Comparaison")


def generate_charts(metrics_list: list[StrategyMetrics], output_path: Path) -> None:
    """
    Génère un graphique à barres comparant les 4 métriques clés.
    """
    import matplotlib.pyplot as plt  # import différé pour accélérer le démarrage

    strategies = [m.strategy_name for m in metrics_list]
    chunk_counts = [m.total_chunks for m in metrics_list]
    index_times = [m.total_indexation_time_seconds for m in metrics_list]
    search_times = [m.average_search_time_seconds for m in metrics_list]
    avg_scores = [m.average_similarity_score for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Comparaison des stratégies de chunking RAG",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["#4C72B0", "#DD8452"]

    axes[0, 0].bar(strategies, chunk_counts, color=colors[: len(strategies)])
    axes[0, 0].set_title("Nombre de chunks")
    axes[0, 0].set_ylabel("Chunks")

    axes[0, 1].bar(strategies, index_times, color=colors[: len(strategies)])
    axes[0, 1].set_title("Temps d'indexation total (s)")
    axes[0, 1].set_ylabel("Secondes")

    axes[1, 0].bar(strategies, search_times, color=colors[: len(strategies)])
    axes[1, 0].set_title("Temps moyen de recherche (s)")
    axes[1, 0].set_ylabel("Secondes")

    axes[1, 1].bar(strategies, avg_scores, color=colors[: len(strategies)])
    axes[1, 1].set_title("Score moyen de similarité")
    axes[1, 1].set_ylabel("Score (0-1)")
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_detailed_retrieval(metrics_list: list[StrategyMetrics]) -> None:
    """Affiche le détail des chunks retournés pour chaque question."""
    for m in metrics_list:
        print(f"\n--- Détail retrieval : {m.strategy_name} ---\n")
        for qr in m.query_results:
            print(f"Question : {qr.question}")
            print(f"  Temps : {qr.search_time_seconds:.4f}s")
            for i, (doc, score) in enumerate(
                zip(qr.documents, qr.similarity_scores), start=1
            ):
                preview = doc[:120].replace("\n", " ") + ("..." if len(doc) > 120 else "")
                print(f"  [{i}] Score={score:.4f} | {preview}")
            print()
