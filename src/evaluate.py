"""
Evaluate the search engine against expert ground truth.

In their comments, process engineers linked some incidents to the same family
(a reference to an earlier incident or to the same analysis sheet). For each
incident in a family, we check whether the engine retrieves another member of
that family among all other incidents, WITHOUT reading the comments (otherwise
the answer would be given to the model).

Metrics:
  Hit@1: the first retrieved incident belongs to the same family
  Hit@5: at least one of the top 5 belongs to the same family
  MRR:   mean of 1 / rank of the first relevant incident (1 = perfect)
  Hit@5 (macro): Hit@5 averaged per family, so each family weighs the same

Usage: python src/evaluate.py [--data path.csv] [--embeddings] [--out prefix]
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from search import IncidentSearch


def metrics_from_ranks(ranks):
    ranks = np.asarray(ranks)
    return {"Hit@1": np.mean(ranks == 1), "Hit@5": np.mean(ranks <= 5), "MRR": np.mean(1 / ranks)}


def first_relevant_ranks(S, families, queries):
    ranks = []
    for i in queries:
        order = np.argsort(-S[i])
        relevant = np.where(families[order] == families[i])[0]
        ranks.append(relevant[0] + 1)
    return np.array(ranks)


def main(path="data/incidents.csv", with_embeddings=False, out="evaluation_results"):
    df = pd.read_csv(path, keep_default_na=False)
    families = df.family.values
    size = df.family.map(df.family.value_counts())
    queries = np.where((families != "") & (size >= 2))[0]
    print(f"{len(df)} incidents, {len(queries)} queries across "
          f"{df.iloc[queries].family.nunique()} families\n")

    results, per_family = {}, {}
    # Baseline: random ranking (averaged over 200 draws)
    rng = np.random.default_rng(0)
    rand = [metrics_from_ranks(first_relevant_ranks(rng.random((len(df), len(df))), families, queries))
            for _ in range(200)]
    results["Random"] = {m: np.mean([r[m] for r in rand]) for m in rand[0]}

    names = {"word": "TF-IDF words", "char": "TF-IDF characters", "hybrid": "Hybrid (words + chars)"}
    if with_embeddings:
        names["embeddings"] = "Multilingual embeddings"
    for method, name in names.items():
        S = IncidentSearch(df, method).self_similarities()
        ranks = first_relevant_ranks(S, families, queries)
        results[name] = metrics_from_ranks(ranks)
        per_family[name] = pd.Series(ranks <= 5, index=families[queries]).groupby(level=0).mean()
        # The largest family holds about half the queries: the macro average
        # gives every family the same weight
        results[name]["Hit@5 (macro)"] = per_family[name].mean()

    tab = pd.DataFrame(results).T.round(2)
    tab.loc["Random", "Hit@5 (macro)"] = np.nan
    print(tab.to_string())
    print("\nHit@5 per family:")
    print(pd.DataFrame(per_family).round(2).to_string())
    tab.to_csv(f"{out}.csv")

    ax = tab.plot.bar(figsize=(9, 4.5), rot=0, ylim=(0, 1.05))
    ax.set(ylabel="Score (1 = perfect)",
           title=f"Retrieving an incident from the same family ({len(queries)} queries)")
    for c, col in zip(ax.containers, tab.columns):
        ax.bar_label(c, labels=["" if pd.isna(v) else f"{v:.2f}" for v in tab[col]], fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out}.png", dpi=150)
    print(f"\nFigure: {out}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/incidents.csv")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--out", default="evaluation_results")
    a = ap.parse_args()
    main(a.data, a.embeddings, a.out)
