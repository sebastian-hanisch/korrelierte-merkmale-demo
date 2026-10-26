"""Auswertung: Einzelwald, Zellen des vorgerechneten Rasters, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import kmk_algorithm as K
import kmk_constants as C
import prs_scenario as S

GRID_PATH = Path(__file__).resolve().parent / "data" / "importance_grid.json"


# --- Einzelwald ---------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Settings:
    world: str = "beide"
    rho: float = C.DEFAULT_RHO
    n: int = C.DEFAULT_N
    trees: int = C.DEFAULT_TREES
    noise: int = C.DEFAULT_NOISE
    threshold: float = C.DEFAULT_THRESHOLD
    seed: int = C.DEFAULT_SEED

    def key(self) -> tuple:
        return (self.world, self.rho, self.n, self.trees, self.noise, self.threshold, self.seed)

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    X: np.ndarray
    y: np.ndarray
    shuffled_a: np.ndarray       # Spalte A nach dem Mischen (Zeilen bleiben bei B)
    singles: np.ndarray          # Wichtigkeit je Spalte (A, B, Rauschen …)
    group: float                 # A und B gemeinsam gemischt
    cond_a: float                # A bedingt auf B
    cond_b: float                # B bedingt auf A
    groups_found: list           # Gruppen aus der Spearman-Korrelation
    outside: float               # Anteil außerhalb des Datenbereichs nach dem Mischen von A (gezählt)
    outside_exact: float
    exact: dict                  # lineares Modell mit bekannten Koeffizienten: "a", "b", "group"


def analyse(s: Settings) -> Analysis:
    data_seed, forest_seed, perm_seed = S.substream_seeds(s.seed, 3)
    X, y = K.draw_twins(s.n, s.rho, s.world, s.noise, S.VRng(data_seed))
    forest = K.fit_forest(X, y, s.trees, forest_seed)
    d = X.shape[1]
    imp = K.oob_importance(forest, X, y, [[j] for j in range(d)] + [[0, 1]], seed=perm_seed)
    cond_a = K.oob_conditional_importance(forest, X, y, 0, 1, seed=perm_seed + 1)
    cond_b = K.oob_conditional_importance(forest, X, y, 1, 0, seed=perm_seed + 2)
    b, cov, sigma = K.twin_model(s.rho, s.world, s.noise)
    exact = {"a": K.linear_importance_exact(b, cov, sigma, [0]), "b": K.linear_importance_exact(b, cov, sigma, [1]), "group": K.linear_importance_exact(b, cov, sigma, [0, 1])}
    return Analysis(s, X, y, K.shuffle_column(X, 0, perm_seed + 3), imp[:d], float(imp[d]), cond_a, cond_b, K.groups_by_spearman(X, s.threshold),
                    K.outside_share(X, 0, 1, s.rho, perm_seed + 3), K.outside_share_exact(s.rho), exact)


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, key: str, rho: float, s: int, extra: int = 0) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, Welt, rho und eine Zusatzzahl)."""
    kinds = ("twin", "linear", "outside", "trees", "probe")
    worlds = list(C.WORLDS) + ["-"]
    return s * 1_000_003 + int(round(rho * 1000)) * 7919 + worlds.index(key) * 1013 + extra * 101 + kinds.index(kind) * 100_000_007


def _sd(values) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def twin_cell(world: str, rho: float, runs_per_seed: int = C.GRID_RUNS, seeds=C.GRID_SEEDS, n: int = C.GRID_N, trees: int = C.GRID_TREES, noise: int = C.GRID_NOISE) -> dict:
    """E1 bis E3 und E5: Wichtigkeit je Spalte, Gruppe und bedingt (Mittel über die Läufe), Aufteilung zwischen A und B (Mittel, Streuung, Anteil extremer Läufe) und der Anteil der Läufe,
    in denen die Spearman-Gruppierung (Schwelle GROUP_THRESHOLD) genau A und B zusammenfasst; daneben die exakten Werte des linearen Modells."""
    rows, shares, found = [], [], []
    for s in seeds:
        rng = S.VRng(cell_seed("twin", world, rho, s))
        for r in range(runs_per_seed):
            X, y = K.draw_twins(n, rho, world, noise, rng)
            sd = s * 1000 + r
            f = K.fit_forest(X, y, trees, sd)
            d = X.shape[1]
            imp = K.oob_importance(f, X, y, [[j] for j in range(d)] + [[0, 1]], seed=sd + 1)
            ca = K.oob_conditional_importance(f, X, y, 0, 1, seed=sd + 2)
            cb = K.oob_conditional_importance(f, X, y, 1, 0, seed=sd + 3)
            rows.append([imp[0], imp[1], float(np.mean(imp[2:d])) if d > 2 else 0.0, imp[d], ca, cb])
            shares.append(K.share(imp[0], imp[1]) if imp[0] + imp[1] > 0 else 0.5)
            found.append(K.groups_by_spearman(X, C.GROUP_THRESHOLD)[0] == [0, 1])
    m = np.mean(rows, axis=0)
    shares = np.array(shares)
    b, cov, sigma = K.twin_model(rho, world, noise)
    return {"world": world, "rho": rho, "runs": len(rows), "a": float(m[0]), "b": float(m[1]), "noise": float(m[2]), "sum": float(m[0] + m[1]), "group": float(m[3]),
            "cond_a": float(m[4]), "cond_b": float(m[5]), "share_mean": float(shares.mean()), "share_sd": _sd(shares), "extreme": float(np.mean((shares < C.EXTREME) | (shares > 1 - C.EXTREME))),
            "found": float(np.mean(found)), "ratio_b_a": float(m[1] / m[0]) if m[0] > 0 else 0.0,
            "exact_a": K.linear_importance_exact(b, cov, sigma, [0]), "exact_b": K.linear_importance_exact(b, cov, sigma, [1]), "exact_group": K.linear_importance_exact(b, cov, sigma, [0, 1])}


def linear_cell(rho: float, runs_per_seed: int = C.GRID_RUNS, seeds=C.GRID_SEEDS, n: int = C.GRID_N, noise: int = C.GRID_NOISE, repeats: int = C.LINEAR_REPEATS) -> dict:
    """E1 und E2: lineares Modell (Welt „beide“) mit angepassten Koeffizienten: Wichtigkeit von A und B, Aufteilung (Mittel, Streuung) und Summe."""
    rows, shares = [], []
    for s in seeds:
        rng = S.VRng(cell_seed("linear", "beide", rho, s))
        for r in range(runs_per_seed):
            X, y = K.draw_twins(n, rho, "beide", noise, rng)
            imp = K.permutation_importance(K.ols_predict(K.fit_ols(X, y)), X, y, [[0], [1]], repeats=repeats, seed=s * 1000 + r)
            rows.append(imp)
            shares.append(K.share(imp[0], imp[1]))
    m = np.mean(rows, axis=0)
    shares = np.array(shares)
    return {"rho": rho, "runs": len(rows), "a": float(m[0]), "b": float(m[1]), "sum": float(m[0] + m[1]), "share_mean": float(shares.mean()), "share_sd": _sd(shares)}


def outside_cell(rho: float, datasets: int = C.OUTSIDE_SETS, n: int = C.GRID_N) -> dict:
    """E4: Anteil der Zeilen außerhalb des 3-Sigma-Bereichs nach dem Mischen von A: gezählt (Mittel über die Datensätze) und exakt."""
    rng = S.VRng(cell_seed("outside", "-", rho, 0))
    vals = []
    for i in range(datasets):
        X, _y = K.draw_twins(n, rho, "beide", 0, rng)
        vals.append(K.outside_share(X, 0, 1, rho, seed=i))
    return {"rho": rho, "datasets": datasets, "measured": float(np.mean(vals)), "exact": K.outside_share_exact(rho)}


def trees_cell(trees: int, rho: float = C.TREE_RHO, datasets: int = C.TREE_DATASETS, forests: int = C.TREE_FORESTS, n: int = C.GRID_N) -> dict:
    """E2: Streuung der Aufteilung A : B (a) bei gleichen Daten und neuen Wäldern (Mittel über `datasets` Datensätze), (b) bei neuen Daten und je einem Wald."""
    sds = []
    for k in range(datasets):
        X, y = K.draw_twins(n, rho, "beide", C.GRID_NOISE, S.VRng(cell_seed("trees", "beide", rho, 900 + k)))
        sh = []
        for r in range(forests):
            f = K.fit_forest(X, y, trees, r)
            imp = K.oob_importance(f, X, y, [[0], [1]], seed=r + 1000)
            sh.append(K.share(imp[0], imp[1]))
        sds.append(_sd(sh))
    rng = S.VRng(cell_seed("trees", "-", rho, 77))
    shb = []
    for r in range(forests):
        X, y = K.draw_twins(n, rho, "beide", C.GRID_NOISE, rng)
        f = K.fit_forest(X, y, trees, 1)
        imp = K.oob_importance(f, X, y, [[0], [1]], seed=2)
        shb.append(K.share(imp[0], imp[1]))
    return {"trees": trees, "rho": rho, "datasets": datasets, "forests": forests, "same_data_sd": float(np.mean(sds)), "new_data_sd": _sd(shb)}


def grid_probe() -> dict:
    """Kleine feste Zellen und ein fester Einzelwald, die der Test nachrechnet: ändert sich Erzeuger oder Kern, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings("beide", 0.9, 200, 8, 1, 0.9, 11))
    return {"twin": twin_cell("beide", 0.9, 2, (7,), 200, 8, 1), "linear": linear_cell(0.9, 2, (7,), 200, 1, 10), "outside_cell": outside_cell(0.9, 5, 200),
            "trees": trees_cell(8, 0.9, 2, 3, 200), "singles": a.singles.tolist(), "group": a.group, "cond_a": a.cond_a, "cond_b": a.cond_b, "found": a.groups_found,
            "outside": a.outside, "exact": a.exact}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f} %".replace(".", ",")


def _num(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _rho(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: wem gehört die Wichtigkeit, wenn A und B (fast) dasselbe messen?"""
    s = a.settings
    ia, ib = float(a.singles[0]), float(a.singles[1])
    noise = a.singles[2:]
    text = (f"Einzeln gemischt: A **{_num(ia)}**, B **{_num(ib)}**" + (f", Rauschmerkmale höchstens {_num(float(np.max(np.abs(noise))))}" if len(noise) else "") + f". "
            f"Gemeinsam gemischt (Gruppe A + B): **{_num(a.group)}**. Bedingt gemischt (jeweils innerhalb von {C.CLASSES} Klassen des Partners): A **{_num(a.cond_a)}**, B **{_num(a.cond_b)}**.")
    if s.world == "nur_a":
        warn = ib > 0.1 * ia
        return ("warning" if warn else "info"), text + (f" Nur A trägt bei, B bekommt trotzdem {_pct(K.share(ib, ia))} der Summe." if warn else " B trägt nichts bei und bekommt (fast) nichts.")
    share_a = K.share(ia, ib) if ia + ib > 0 else 0.5
    extreme = share_a < C.EXTREME or share_a > 1 - C.EXTREME
    text += f" Aufteilung A : B = {_pct(share_a)} : {_pct(1 - share_a)}."
    return ("warning" if extreme else "info"), text + (" **Einer der beiden Zwillinge bekommt fast alles**: mit neuen Daten wäre es anders." if extreme else "")


def verdict_exact(a: Analysis) -> tuple[str, str]:
    """Lineares Modell mit bekannten Koeffizienten: exakte Wichtigkeit, zum Vergleich mit dem Wald."""
    e = a.exact
    return "info", (f"Zur Gegenprobe: ein lineares Modell mit den **wahren** Koeffizienten hätte exakt A **{_num(e['a'])}**, B **{_num(e['b'])}**, Gruppe **{_num(e['group'])}** "
                    f"(Summe der Einzelwerte {_num(e['a'] + e['b'])}); die Aufteilung wäre {'50 : 50' if abs(e['a'] - e['b']) < 1e-9 else _pct(K.share(e['a'], e['b'])) + ' : ' + _pct(K.share(e['b'], e['a']))}.")


def verdict_outside(a: Analysis) -> tuple[str, str]:
    """Extrapolation: Anteil der Zeilen außerhalb des Datenbereichs nach dem Mischen von A."""
    text = (f"Nach dem Mischen von A liegen **{_pct(a.outside)}** der Zeilen außerhalb des {C.K_SIGMA:g}-Sigma-Bereichs der gemeinsamen Verteilung von A und B (exakt: {_pct(a.outside_exact)}); "
            f"bei unabhängigen Merkmalen wären es {_pct(K.outside_share_exact(0.0))}.")
    return ("warning" if a.outside_exact > 0.2 else "info"), text + (" Diese Zeilen liegen in einem Bereich, in dem nur etwa 1 % der echten Zeilen liegen; vermutlich extrapoliert der Wald dort." if a.outside_exact > 0.2 else "")


def verdict_groups(a: Analysis) -> tuple[str, str]:
    """Gruppen aus der Spearman-Korrelation bei der gewählten Schwelle."""
    s = a.settings
    names = lambda g: " + ".join(("A", "B")[j] if j < 2 else f"Rauschen {j - 1}" for j in g)          # noqa: E731
    pairs = [g for g in a.groups_found if len(g) > 1]
    if any(set(g) >= {0, 1} for g in pairs):
        return "success", f"Bei der Schwelle |Spearman| ≥ {_num(s.threshold)} bilden **A und B eine Gruppe** ({'; '.join(names(g) for g in pairs)}): gemeinsam gemischt hat sie die Wichtigkeit **{_num(a.group)}**."
    return "warning", f"Bei der Schwelle |Spearman| ≥ {_num(s.threshold)} findet die Gruppierung **keine** Gruppe aus A und B; die Zwillinge bleiben einzeln (gemeinsam gemischt wären sie {_num(a.group)})."


def verdict_sum_grid(rows: list[dict], linear: list[dict]) -> tuple[str, str]:
    """E1: Summe der Einzelwichtigkeiten gegen rho bei der Welt „beide“, mit Gruppe und exakter Summe des linearen Modells."""
    cells = sorted((r for r in rows if r["world"] == "beide"), key=lambda r: r["rho"])
    first, mid, last = cells[0], find(cells, rho=0.5) or cells[len(cells) // 2], cells[-1]
    return "warning", (f"Beide tragen bei: die Summe der Einzelwichtigkeiten ist **{_num(first['sum'])}** bei ρ = {_rho(first['rho'])}, **{_num(mid['sum'])}** bei ρ = {_rho(mid['rho'])} und {_num(last['sum'])} bei ρ = {_rho(last['rho'])}; "
                       f"die Gruppe bleibt bei {_num(mid['group'])} bis {_num(last['group'])} (ρ = 0: {_num(first['group'])}). Im linearen Modell mit bekannten Koeffizienten wäre die Summe bei ρ = {_rho(mid['rho'])} exakt {_num(mid['exact_a'] + mid['exact_b'])}.")


def verdict_spread(rows: list[dict], linear: list[dict], rho: float) -> tuple[str, str]:
    """E2: Streuung der Aufteilung bei einem rho (Wald und angepasstes lineares Modell)."""
    c = find([r for r in rows if r["world"] == "beide"], rho=rho)
    lin = find(linear, rho=rho)
    return ("warning" if c["share_sd"] > 0.1 else "info"), (f"ρ = {_rho(rho)}: der Anteil von A schwankt von Datensatz zu Datensatz mit der Streuung **{_num(c['share_sd'], 3)}** (Mittel {_num(c['share_mean'])}); "
                                                          f"in **{_pct(c['extreme'])}** der {c['runs']} Läufe gehört einem Zwilling mehr als {_pct(1 - C.EXTREME)}. Das angepasste lineare Modell streut mit {_num(lin['share_sd'], 3)}.")


def verdict_trees(rows: list[dict]) -> tuple[str, str]:
    """E2: Zahl der Bäume gegen die Streuung der Aufteilung bei gleichen und bei neuen Daten."""
    rows = sorted(rows, key=lambda r: r["trees"])
    lo, hi = rows[0], rows[-1]
    return "warning", (f"ρ = {_rho(lo['rho'])}: mit {lo['trees']} Bäumen und gleichen Daten streut die Aufteilung mit {_num(lo['same_data_sd'], 3)}, mit {hi['trees']} Bäumen nur noch mit {_num(hi['same_data_sd'], 3)}; "
                       f"bei **neuen Daten** bleibt sie bei {_num(lo['new_data_sd'], 3)} und {_num(hi['new_data_sd'], 3)}: mehr Bäume beruhigen sie kaum, sie hängt an der Stichprobe.")


def verdict_only_a(rows: list[dict]) -> tuple[str, str]:
    """E3: B ohne eigene Information: Wichtigkeit von A und B gegen rho."""
    cells = sorted((r for r in rows if r["world"] == "nur_a"), key=lambda r: r["rho"])
    calm = [r for r in cells if r["rho"] <= 0.9]
    top = cells[-1]
    return "warning", (f"Nur A trägt bei: bis ρ = {_rho(calm[-1]['rho'])} bleibt B bei höchstens {_num(max(r['b'] for r in calm), 3)}; bei ρ = {_rho(top['rho'])} bekommt B **{_num(top['b'])}** "
                       f"und A fällt von {_num(cells[0]['a'])} auf {_num(top['a'])}. Bis ρ = {_rho(calm[-1]['rho'])} gilt die Verwässerung nur, wenn beide Zwillinge Information tragen; ab ρ = 0,99 verteilt der Wald die Wichtigkeit auch auf eine bloße Kopie.")


def verdict_outside_grid(rows: list[dict], rho: float) -> tuple[str, str]:
    """E4: gezählter und exakter Anteil außerhalb bei einem rho."""
    c = find(rows, rho=rho)
    return ("warning" if c["exact"] > 0.2 else "info"), f"ρ = {_rho(rho)}: nach dem Mischen von A liegen gezählt **{_pct(c['measured'])}** der Zeilen außerhalb des {C.K_SIGMA:g}-Sigma-Bereichs, exakt {_pct(c['exact'])}."


def verdict_remedies(rows: list[dict], rho: float) -> tuple[str, str]:
    """E5: drei Antworten bei einem rho: einzeln, Gruppe, bedingt; und ob die Spearman-Gruppierung die Gruppe findet."""
    c = find([r for r in rows if r["world"] == "beide"], rho=rho)
    return "info", (f"ρ = {_rho(rho)}: einzeln gemischt A **{_num(c['a'])}** und B **{_num(c['b'])}**, als Gruppe **{_num(c['group'])}**, bedingt auf den Partner A **{_num(c['cond_a'])}** und B **{_num(c['cond_b'])}**. "
                    f"Die Spearman-Gruppierung (Schwelle {_num(C.GROUP_THRESHOLD)}) fasst A und B in {_pct(c['found'])} der Läufe zusammen.")
