"""
Extract and clean an incident log stored in an Excel workbook.

Everything specific to the source log (sheet name, column positions, incident ID
format, cross-reference patterns) is read from a local config file that is
excluded from the repository:

    cp config.example.json config.local.json   # then edit the values
    python src/prepare.py path/to/incident_log.xlsx

Output: data/incidents.csv (excluded from the repository). Personal columns
(signatures, owners) are dropped and incident families are renamed F1, F2...
"""
import json
import re
import sys
from pathlib import Path

import openpyxl
import pandas as pd


def load_config():
    for name in ("config.local.json", "config.example.json"):
        if Path(name).exists():
            with open(name, encoding="utf-8") as f:
                return json.load(f)
    raise SystemExit("No configuration file found.")


def clean_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).split())


def key(text):
    """Comparison key, insensitive to case and accents."""
    return text.upper().replace("É", "E").replace("é", "E")


def families(df, cfg):
    """Incident families derived from cross-references written in the comments.

    Used only as ground truth for evaluation: the search engine never reads
    the comments.
    """
    pattern = re.compile(cfg["cross_ref_pattern"], re.I)
    prefix = cfg["family_comment_prefix"].lower()
    cited_ids = {key(m) for c in df.comment for m in pattern.findall(c)}
    raw = []
    for comment, ident in zip(df.comment, df.id):
        refs = pattern.findall(comment)
        if refs:
            raw.append(key(refs[0]))
        elif prefix and comment.lower().startswith(prefix):
            raw.append("SHARED_COMMENT")
        elif key(ident) in cited_ids:
            raw.append(key(ident))
        else:
            raw.append("")
    # Anonymisation: F1 = largest family, F2 = next one...
    s = pd.Series(raw)
    order = s[s != ""].value_counts().index
    return s.map({f: f"F{i + 1}" for i, f in enumerate(order)}).fillna("")


def prepare(path, cfg):
    ws = openpyxl.load_workbook(path, data_only=True)[cfg["sheet"]]
    id_pattern = re.compile(cfg["id_pattern"])
    rows = [r for r in ws.iter_rows(min_row=cfg["first_row"], values_only=True)
            if r[0] and id_pattern.match(str(r[0]))]
    df = pd.DataFrame([{name: r[i] for name, i in cfg["columns"].items()} for r in rows])

    for c in df.columns:
        if c != "date":
            df[c] = df[c].map(clean_text)
    df["category"] = df.category.str.capitalize()  # harmonise inconsistent casing

    n_before = len(df)
    df = df.drop_duplicates(subset=["id", "description"]).reset_index(drop=True)
    df["family"] = families(df, cfg)
    return df, n_before - len(df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/prepare.py path/to/incident_log.xlsx")
    cfg = load_config()
    df, n_duplicates = prepare(sys.argv[1], cfg)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/incidents.csv", index=False)  # read back with keep_default_na=False

    print(f"{len(df)} incidents kept ({n_duplicates} exact duplicate(s) removed)")
    print(f"Resolution filled in: {(df.resolution.str.len() > 0).sum()}")
    fam = df[df.family != ""].family.value_counts()
    print(f"Families: {len(fam)} ({fam[fam >= 2].sum()} incidents in families of 2 or more)")
    n_shared_ids = df.id.duplicated(keep=False).sum()
    if n_shared_ids:
        print(f"To check: {n_shared_ids} rows share an incident ID with another row")
