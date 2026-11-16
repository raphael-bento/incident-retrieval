"""
Generate an ENTIRELY FICTIONAL incident log, in the same format as
data/incidents.csv, so the repository runs without any real data.

Scenarios, equipment, categories and products are invented and unrelated to the
real log: no text is copied or paraphrased from it. Descriptions are written in
French to match the language of the real log.

Scores obtained on this log only check that the code runs. They are NOT a measure
of real-world performance: incidents of a fictional family are generated from the
same sentence templates, so they are almost identical by construction.

Usage: python src/generate_demo.py [--n-families 12] [--n-noise 150] [--seed 0]
Output: demo/fictional_incident_log.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Recurring scenarios: each one becomes an incident family.
# Several phrasings per scenario, as if written by different operators.
# --------------------------------------------------------------------------
FAMILIES = [
    {"zone": "Presse P1", "category": "Équipement", "detail": "Défaillance",
     "severity": "Elevée", "resolution": "Remplacement du flexible hydraulique et contrôle hebdomadaire ajouté.",
     "templates": ["Fuite d'huile hydraulique sous la presse P1, flaque d'environ {n} cm.",
                "Presse P1 : écoulement d'huile au niveau du vérin principal.",
                "Traces d'huile sur le sol autour de la presse, flexible suintant.",
                "Nouvelle fuite hydraulique sur P1, production arrêtée {n} min."]},
    {"zone": "Cuve de stockage C2", "category": "Équipement", "detail": "Défaillance",
     "severity": "Modérée", "resolution": "Nettoyage puis remplacement du capteur de niveau.",
     "templates": ["Alarme niveau haut sur la cuve C2 alors que la cuve est à moitié vide.",
                "Fausse alarme du capteur de niveau de C2, remplissage bloqué.",
                "Capteur niveau C2 en défaut, indication incohérente avec le niveau réel.",
                "Encore une alarme niveau haut intempestive sur C2."]},
    {"zone": "Visseuse V1", "category": "Équipement", "detail": "Défaillance",
     "severity": "Modérée", "resolution": "Ajout d'un guide en entrée de visseuse.",
     "templates": ["Bourrage de pots à l'entrée de la visseuse, {n} arrêts dans le poste.",
                "Les pots se coincent dans le guide d'entrée de la visseuse.",
                "Accumulation de pots en entrée V1, convoyeur à l'arrêt.",
                "Blocage récurrent des pots avant la visseuse, intervention manuelle."]},
    {"zone": "Visseuse V1", "category": "Produit", "detail": "Hors tolérance",
     "severity": "Elevée", "resolution": "Réétalonnage du couple de serrage de la tête de vissage.",
     "templates": ["Couple de serrage insuffisant sur {n} couvercles, contrôle non conforme.",
                "Couvercles mal vissés détectés au contrôle final.",
                "Bouchons qui tournent à la main sur une partie du lot {lot}.",
                "Serrage trop faible des couvercles, {n} pots à revisser."]},
    {"zone": "Étiqueteuse E1", "category": "Produit", "detail": "Aspect",
     "severity": "Basse", "resolution": "Réglage de la position du capteur de détection d'étiquette.",
     "templates": ["Étiquettes posées de travers sur {n} pots, décalage de quelques mm.",
                "Mauvais positionnement des étiquettes, plis sur le bord.",
                "Étiquette décalée ou froissée sur une partie du lot {lot}.",
                "Pose étiquette pas droite, {n} pots à rétiqueter."]},
    {"zone": "Contrôle final", "category": "Produit", "detail": "Aspect",
     "severity": "Modérée", "resolution": "Remplacement du ruban de l'imprimante transfert thermique.",
     "templates": ["Code-barres illisible au lecteur sur {n} cartons.",
                "Le scanner ne lit pas les codes-barres du lot {lot}, impression trop pâle.",
                "Impression des codes-barres effacée, rejet au contrôle final.",
                "Codes-barres pâles, {n} cartons ré-étiquetés."]},
    {"zone": "Four F1", "category": "Équipement", "detail": "Hors tolérance",
     "severity": "Modérée", "resolution": "Recalibrage de la sonde de température du four.",
     "templates": ["Écart de {n}°C entre la sonde du four et le thermomètre de contrôle.",
                "Sonde four F1 décalée par rapport à la mesure de référence.",
                "Mesure de température du four non cohérente avec le contrôle manuel."]},
    {"zone": "Réseau air comprimé", "category": "Équipement", "detail": "Défaillance",
     "severity": "Elevée", "resolution": "Réparation de la fuite sur le réseau et réglage du compresseur.",
     "templates": ["Pression d'air insuffisante, vérins lents sur toute la ligne.",
                "Chute de pression du réseau d'air à {n} bar, machines en défaut.",
                "Manque d'air comprimé en milieu de poste, arrêt de la ligne.",
                "Pression air trop basse, alarmes sur plusieurs machines."]},
    {"zone": "Poste de pesée", "category": "Documentation", "detail": "Information erronée",
     "severity": "Basse", "resolution": "Procédure obsolète retirée du poste et version en vigueur affichée.",
     "templates": ["Ancienne version de la procédure de pesée affichée au poste.",
                "La procédure au poste n'est pas la version en vigueur (indice {n}).",
                "Mauvais indice de procédure affiché en salle de pesée."]},
    {"zone": "Poste de pesée", "category": "Autre", "detail": "Hors tolérance",
     "severity": "Modérée", "resolution": "Révision du déshumidificateur de la salle.",
     "templates": ["Humidité relative de la salle de pesée à {n} %, au-dessus de la limite.",
                "Hygrométrie trop élevée en salle de pesée, pesées reportées.",
                "Taux d'humidité hors limite dans la salle, déshumidificateur à l'arrêt."]},
    {"zone": "Pompe de transfert T1", "category": "Équipement", "detail": "Défaillance",
     "severity": "Modérée", "resolution": "Remplacement du filtre et ajout au plan de maintenance préventive.",
     "templates": ["Débit de transfert anormalement faible, filtre de la pompe T1 colmaté.",
                "Transfert très lent entre cuves, {n} min au lieu de 15.",
                "Pompe T1 : filtre bouché, débit réduit.",
                "Filtre colmaté sur la ligne de transfert, nettoyage effectué."]},
    {"zone": "Doseuse D1", "category": "Produit", "detail": "Hors tolérance",
     "severity": "Modérée", "resolution": "Recalage de la course de la pompe doseuse.",
     "templates": ["Masse dosée hors tolérance sur {n} pots, sous-dosage.",
                "Pesée de contrôle : {n} pots en dessous du minimum.",
                "Dose instable en début de production, pots trop légers.",
                "Sous-dosage récurrent sur la doseuse, réglage pompe à revoir."]},
]

# One-off incidents (noise): distractors that belong to no family
NOISE = [
    ("Magasin", "Documentation", "Information erronée", "Code article de l'emballage absent de la nomenclature."),
    ("Atelier", "Informatique", "Autre", "Impression des étiquettes impossible, imprimante hors ligne."),
    ("Presse P1", "Documentation", "Information erronée", "Temps de cycle de l'étape {e} absent de la fiche."),
    ("Four F1", "Équipement", "Défaillance", "Porte du four difficile à fermer, joint usé."),
    ("Étiqueteuse E1", "Équipement", "Matériel manquant", "Rouleau d'étiquettes de rechange introuvable."),
    ("Pompe de transfert T1", "Documentation", "Information erronée", "Débit nominal indiqué hors plage de la pompe."),
    ("Visseuse V1", "Équipement", "Défaillance", "Bruit anormal du moteur de la visseuse."),
    ("Magasin", "Produit", "Aspect", "Emballage carton abîmé à réception, lot {lot}."),
    ("Doseuse D1", "Équipement", "Matériel manquant", "Buse de rechange de diamètre {d} mm manquante."),
    ("Atelier", "Autre", "Autre", "Balance de contrôle hors date d'étalonnage."),
    ("Cuve de stockage C2", "Documentation", "Information erronée", "Ordre de nettoyage des cuves inversé entre deux pages."),
    ("Atelier", "Produit", "Corps étranger", "Corps étranger trouvé dans un contenant, lot {lot}."),
    ("Four F1", "Documentation", "Information erronée", "Durée de cuisson différente entre la fiche et le planning."),
    ("Étiqueteuse E1", "Documentation", "Information erronée", "Date de péremption erronée sur le modèle d'étiquette."),
    ("Magasin", "Autre", "Autre", "Stock de pots vides insuffisant pour l'ordre de fabrication prévu."),
]

TYPOS = {"température": ["temp", "T°", "temperature"], "production": ["prod"],
          "pression": ["press."], "étiquettes": ["etiquettes", "étiquètes"],
          "couvercles": ["couv."], "humidité": ["humidite", "HR"]}


def fill(template, rng):
    return template.format(e=rng.randint(2, 14), n=rng.randint(2, 40),
                           lot=f"PX-{rng.randint(1000, 9999)}", d=rng.choice([4, 6, 8]))


def add_noise(text, rng, p_abbrev=0.5, p_typo=0.05):
    """Mimic operator input: abbreviations and occasional typos."""
    for word, variants in TYPOS.items():
        if word in text and rng.random() < p_abbrev:
            text = text.replace(word, rng.choice(variants))
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < p_typo:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # swap two adjacent letters
    return " ".join(words)


def generate(n_families=12, n_noise=150, seed=0):
    rng = random.Random(seed)
    rows = []
    dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")

    for k, fam in enumerate(FAMILIES[:n_families]):
        for _ in range(rng.randint(3, 12)):
            rows.append({"zone": fam["zone"], "category": fam["category"], "detail": fam["detail"],
                         "description": add_noise(fill(rng.choice(fam["templates"]), rng), rng),
                         "severity": fam["severity"], "resolution": fam["resolution"],
                         "family": f"F{k + 1}"})
    for _ in range(n_noise):
        zone, category, detail, template = rng.choice(NOISE)
        rows.append({"zone": zone, "category": category, "detail": detail,
                     "description": add_noise(fill(template, rng), rng),
                     "severity": rng.choice(["Basse", "Modérée", ""]), "resolution": "",
                     "family": ""})

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"DEMO-{i + 1:04d}" for i in range(len(df))])
    df.insert(1, "date", sorted(rng.sample(list(dates), len(df))))
    df["comment"] = ""
    df["status"] = "Closed"
    df["redundant"] = ""
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-families", type=int, default=12)
    ap.add_argument("--n-noise", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = generate(a.n_families, a.n_noise, a.seed)
    Path("demo").mkdir(exist_ok=True)
    df.to_csv("demo/fictional_incident_log.csv", index=False)
    print(f"FICTIONAL log: {len(df)} incidents, {df[df.family != ''].family.nunique()} families "
          f"-> demo/fictional_incident_log.csv")
