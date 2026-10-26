"""Konstanten: Regler-Grenzen, Welten, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Spearman kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py), der Wald-Kern als wörtliche Kopie aus
random-forest-demo (rf_tree.py, rf_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): ein Wald mit n = 1 500 Zeilen, 60 Bäumen und 8 Kennzahlen braucht für die Anpassung etwa eine bis eineinhalb Sekunden, die Wichtigkeiten
(einzeln, Gruppe, bedingt) je Baum einen kurzen Durchlauf über die Out-of-Bag-Zeilen. Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/importance_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

RHOS = (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999)
DEFAULT_RHO = 0.5
N_MIN, N_MAX, DEFAULT_N = 200, 1500, 600
TREES_MIN, TREES_MAX, DEFAULT_TREES = 10, 60, 30
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 4, 2
THRESHOLD_MIN, THRESHOLD_MAX, DEFAULT_THRESHOLD = 0.7, 0.99, 0.9
DEFAULT_SEED = 35

MIN_LEAF = 5                 # kleinste Blattgröße der Bäume
MIN_OOB = 10                 # Bäume mit weniger Out-of-Bag-Zeilen zählen nicht
CLASSES = 10                 # Zahl der Quantilklassen beim bedingten Mischen
NOISE_SD = 0.5               # Standardabweichung des Rauschens der Zielgröße
K_SIGMA = 3.0                # Radius des Bereichs „es gibt Zeilen wie diese“ (Mahalanobis-Abstand der Zwillinge)
EXTREME = 0.3                # Aufteilung außerhalb von EXTREME : 1 − EXTREME zählt als „einer gehört fast alles“

# Welten: Schlüssel -> Anzeigename
WORLDS = {
    "beide": "Beide tragen bei (Wartezeit = A + B + Rauschen)",
    "nur_a": "Nur A trägt bei (B ist eine verrauschte Kopie)",
}
WORLD_SHORT = {"beide": "beide tragen bei", "nur_a": "nur A trägt bei"}

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
GRID_RHOS = (0.0, 0.5, 0.9, 0.99, 0.999)
GRID_N, GRID_TREES, GRID_NOISE, GRID_RUNS = 600, 30, 2, 25          # Läufe je Seed
OUTSIDE_SETS = 200                                                   # Datensätze je Zelle der Extrapolation
LINEAR_REPEATS = 200                                                 # Umordnungen je Wichtigkeit im linearen Modell
TREE_COUNTS, TREE_RHO, TREE_DATASETS, TREE_FORESTS = (10, 30, 100), 0.99, 6, 40
GROUP_THRESHOLD = 0.9                                                # Schwelle der Gruppenbildung im Raster

COLOR_A = "#4c78a8"
COLOR_B = "#e45756"
COLOR_NOISE = "#9d9d9d"
COLOR_GROUP = "#54a24b"
COLOR_COND = "#f58518"
COLOR_EDGE = "#222222"
COLOR_FOREST = "#4c78a8"
COLOR_LINEAR = "#f58518"

PRESETS = {
    "Standardfall (Voreinstellung)": {"world": "beide", "rho": 0.5, "n": 600, "trees": 30, "noise": 2, "threshold": 0.9, "seed": DEFAULT_SEED},
    "Dieselbe Information, springende Aufteilung": {"world": "beide", "rho": 0.99, "n": 600, "trees": 30, "noise": 2, "threshold": 0.9, "seed": DEFAULT_SEED},
    "Nur A trägt bei": {"world": "nur_a", "rho": 0.999, "n": 600, "trees": 30, "noise": 2, "threshold": 0.9, "seed": DEFAULT_SEED},
    "Ohne Zwilling": {"world": "beide", "rho": 0.0, "n": 600, "trees": 30, "noise": 2, "threshold": 0.9, "seed": DEFAULT_SEED},
    "Gruppe bilden": {"world": "beide", "rho": 0.9, "n": 600, "trees": 30, "noise": 2, "threshold": 0.8, "seed": DEFAULT_SEED},
    "Mischen erzeugt Unmögliches": {"world": "beide", "rho": 0.99, "n": 300, "trees": 30, "noise": 2, "threshold": 0.9, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen Zahlen des vorgerechneten Rasters (data/importance_grid.json) und exakte Werte; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Beide tragen bei, ρ = 0,5: Im Raster ist die Summe der Einzelwichtigkeiten 1,23 (bei ρ = 0: 1,67), die Gruppe aus A und B 1,77 und die bedingte Wichtigkeit je Zwilling 0,44.",
    "Dieselbe Information, springende Aufteilung": "ρ = 0,99, beide tragen bei: Im Raster (100 Wälder) bleibt der mittlere Anteil von A bei 49,0 %, doch er streut mit 0,181 von Datensatz zu Datensatz, und in etwa einem Drittel der Läufe (33,0 % bei 100 Wäldern) gehört einem Zwilling mehr als 70 %.",
    "Nur A trägt bei": "Nur A trägt bei, ρ = 0,999: Im Raster bekommt B 0,50 und A nur noch 0,99 (bei ρ = 0: A 1,87, B 0,00); bis ρ = 0,9 blieb B bei höchstens 0,002.",
    "Ohne Zwilling": "Unabhängige Zwillinge, beide tragen bei: A 0,83, B 0,84, die Summe 1,67 und die Gruppe 1,67 sind gleich; bedingt 0,80 und 0,81; Rauschmerkmale 0,00 (Raster).",
    "Gruppe bilden": "ρ = 0,9: Gemeinsam gemischt hat die Gruppe 1,84, die Summe der Einzelwerte nur 1,22. Die Schwelle der Gruppenbildung ist ein Regler: bei 0,9 findet die Spearman-Gruppierung das Paar im Raster nur in 17,0 % der Läufe.",
    "Mischen erzeugt Unmögliches": "ρ = 0,99: Nach dem Mischen von A liegen exakt 77,1 % der Zeilen außerhalb des 3-Sigma-Bereichs (bei unabhängigen Merkmalen 1,1 %): Zeilen aus einem Bereich, in dem nur etwa 1 % der echten Zeilen liegen.",
}
