"""
Robustness test: does the engine still find the right family when the query
is poorly typed?

Each query description is degraded (typos, abbreviations, truncation, missing
categories) and Hit@5 is compared with the original query. Incidents in the
index are never modified: only the user's input is degraded, as in real use.

The abbreviation list targets French, the language of the source log.

Usage: python src/robustness_test.py [--data data/incidents.csv]
Output: robustness_results.csv
"""
import argparse
import random

import numpy as np
import pandas as pd

from search import IncidentSearch

ABBREVIATIONS = {"température": "T°", "temperature": "T°", "maximum": "max", "minimum": "min",
                 "production": "prod", "minutes": "min", "fabrication": "fab", "contrôle": "ctrl"}


def typos(text, rng, p=0.15):
    """Swap two adjacent letters in about 15% of words longer than 4 letters."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < p:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def abbreviate(text, rng):
    for word, abbr in ABBREVIATIONS.items():
        text = text.replace(word, abbr).replace(word.capitalize(), abbr)
    return text


def truncate(text, rng, n=6):
    return " ".join(text.split()[:n])


DEGRADATIONS = {
    "Original query": (lambda t, rng: t, True),
    "Typos": (typos, True),
    "Abbreviations": (abbreviate, True),
    "Truncated description (6 words)": (truncate, True),
    "Description only (no zone or category)": (lambda t, rng: t, False),
}


def hit5(engine, df, queries, degrade, with_categories, seed):
    rng = random.Random(seed)
    families = df.family.values
    texts = []
    for i in queries:
        desc = degrade(df.description[i], rng)
        texts.append(f"{df.zone[i]} | {df.category[i]} | {df.detail[i]} | {desc}"
                     if with_categories else desc)
    S = engine.similarities(texts)
    hits = []
    for row, i in zip(S, queries):
        row = row.copy()
        row[i] = -np.inf  # exclude the incident itself
        top5 = np.argsort(-row)[:5]
        hits.append(np.any(families[top5] == families[i]))
    return np.mean(hits)


def main(path, n_draws=20):
    df = pd.read_csv(path, keep_default_na=False)
    size = df.family.map(df.family.value_counts())
    queries = np.where((df.family != "") & (size >= 2))[0]
    print(f"{len(queries)} queries, mean Hit@5 over {n_draws} random draws\n")

    results = {}
    for method, name in {"word": "TF-IDF words", "char": "TF-IDF characters",
                         "hybrid": "Hybrid"}.items():
        engine = IncidentSearch(df, method)
        results[name] = {
            deg: np.mean([hit5(engine, df, queries, f, cat, s) for s in range(n_draws)])
            for deg, (f, cat) in DEGRADATIONS.items()
        }
    tab = pd.DataFrame(results).round(2)
    print(tab.to_string())
    tab.to_csv("robustness_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/incidents.csv")
    main(ap.parse_args().data)
