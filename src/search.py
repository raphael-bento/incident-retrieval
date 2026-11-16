"""
Similar-incident search engine.

Each incident is represented by the text available WHEN IT IS REPORTED
(zone, category, detail, description). Comments and resolutions, written later,
are never used for retrieval: they are only displayed.

The source log is written in French, so text normalisation targets French
(accents, the "T°" shorthand for temperature).

Usage:
  python src/search.py "dépassement de température à l'étape 8" --k 5
  python src/search.py "pression air basse" --data demo/fictional_incident_log.csv
"""
import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METHODS = ["word", "char", "hybrid", "embeddings"]


def normalize(text):
    """Lowercase, strip accents, expand 'T°' to 'temperature'."""
    text = text.lower().replace("t°c", " temperature ").replace("t°", " temperature ")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def report_text(df):
    return df.zone + " | " + df.category + " | " + df.detail + " | " + df.description


class IncidentSearch:
    def __init__(self, df, method="hybrid", embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.df = df.reset_index(drop=True)
        self.method = method
        corpus = report_text(self.df).map(normalize)

        if method in ("word", "hybrid"):
            # TF-IDF on words and word pairs: captures domain vocabulary
            self.vec_word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.X_word = self.vec_word.fit_transform(corpus)
        if method in ("char", "hybrid"):
            # TF-IDF on 3- to 5-character fragments: robust to typos
            self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.X_char = self.vec_char.fit_transform(corpus)
        if method == "embeddings":
            # Optional: requires `pip install sentence-transformers` and internet access
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            self.X_emb = self.model.encode(list(corpus), normalize_embeddings=True)

    def similarities(self, queries):
        """(n_queries × n_incidents) cosine similarity matrix."""
        queries = [normalize(q) for q in queries]
        if self.method == "word":
            return cosine_similarity(self.vec_word.transform(queries), self.X_word)
        if self.method == "char":
            return cosine_similarity(self.vec_char.transform(queries), self.X_char)
        if self.method == "hybrid":
            return 0.5 * (cosine_similarity(self.vec_word.transform(queries), self.X_word)
                          + cosine_similarity(self.vec_char.transform(queries), self.X_char))
        q = self.model.encode(queries, normalize_embeddings=True)
        return q @ self.X_emb.T

    def self_similarities(self):
        """Similarity of each incident to all others (diagonal excluded)."""
        S = self.similarities(list(report_text(self.df)))
        np.fill_diagonal(S, -np.inf)
        return S

    def search(self, query, k=5):
        s = self.similarities([query])[0]
        top = np.argsort(-s)[:k]
        res = self.df.iloc[top][["id", "date", "zone", "description", "resolution"]].copy()
        res.insert(0, "score", s[top].round(3))
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="hybrid", choices=METHODS)
    ap.add_argument("--data", default="data/incidents.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.data, keep_default_na=False)
    engine = IncidentSearch(df, a.method)
    for _, r in engine.search(a.query, a.k).iterrows():
        print(f"\n[{r.score:.2f}] {r.id} — {str(r.date)[:10]} — {r.zone}")
        print(f"  Description: {r.description[:200]}")
        print(f"  Resolution:  {r.resolution[:200] or '(not filled in)'}")
