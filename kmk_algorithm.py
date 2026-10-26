"""Kern: Zwillingsmerkmale, Permutationswichtigkeit (Out-of-Bag im Wald, allgemein für jedes Modell), Gruppenmischen, bedingtes Mischen in Klassen, Anteil außerhalb des Datenbereichs
(exakt und durch Zählen), Gruppen aus der Spearman-Korrelation, lineares Modell mit exakter Formel.

Wichtigkeit = Anstieg des mittleren Fehlerquadrats beim Mischen, geteilt durch Var(y), also der Abfall von R². Der Wald-Kern kommt als wörtliche Kopie aus random-forest-demo
(rf_tree.py, rf_algorithm.py), Zufall und Spearman aus pearson-demo (prs_*.py). Ohne scipy zur Laufzeit."""

import math

import numpy as np

import kmk_constants as C
import rf_algorithm as RF
import rf_tree as T
from prs_algorithm import spearman_r

_erfc = np.frompyfunc(math.erfc, 1, 1)


# --- Daten -------------------------------------------------------------------------------------------------------------------------------

def twin_model(rho: float, world: str, n_noise: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Wahre Koeffizienten b, Kovarianzmatrix der Merkmale und Standardabweichung des Rauschens der Zielgröße. Spalten: A, B, dann n_noise unabhängige Rauschmerkmale.
    Welt "beide": y = A + B + 0,5 e. Welt "nur_a": y = 2 A + 0,5 e (B ist nur eine verrauschte Kopie von A)."""
    if world not in C.WORLDS:
        raise ValueError(f"unbekannte Welt: {world}")
    d = 2 + n_noise
    cov = np.eye(d)
    cov[0, 1] = cov[1, 0] = rho
    b = np.zeros(d)
    if world == "beide":
        b[0] = b[1] = 1.0
    else:
        b[0] = 2.0
    return b, cov, C.NOISE_SD


def draw_twins(n: int, rho: float, world: str, n_noise: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n Zeilen: A und B (Standardnormal, Korrelation rho), n_noise unabhängige Rauschmerkmale und die Zielgröße y."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho muss in [0, 1) liegen")
    b, _cov, sigma = twin_model(rho, world, n_noise)
    z1, z2 = rng.normal(n), rng.normal(n)
    cols = [z1, rho * z1 + math.sqrt(1.0 - rho * rho) * z2] + [rng.normal(n) for _ in range(n_noise)]
    X = np.column_stack(cols)
    return X, X @ b + sigma * rng.normal(n)


def fit_forest(X, y, trees: int, seed: int) -> RF.Forest:
    """Wald aus `trees` vollen Regressionsbäumen auf Bootstrap-Stichproben (alle Merkmale an jedem Knoten)."""
    return RF.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float), "reg", min_leaf=C.MIN_LEAF, n_trees=trees, mtry=None, seed=seed)


# --- Wichtigkeit -------------------------------------------------------------------------------------------------------------------------

def _oob_trees(forest: RF.Forest, X, y, min_oob: int):
    """Je Baum: Baum, Out-of-Bag-Zeilen der Merkmale und der Zielgröße (nur Bäume mit mindestens min_oob solchen Zeilen)."""
    for b, tree in enumerate(forest.trees):
        oob = np.nonzero(~forest.in_bag[b])[0]
        if len(oob) >= min_oob:
            yield tree, X[oob], y[oob]


def _mse(tree, X, y) -> float:
    return float(np.mean((y - T.predict_value(tree, X)) ** 2))


def oob_importance(forest: RF.Forest, X, y, groups, seed: int = 0, min_oob: int = C.MIN_OOB) -> np.ndarray:
    """Wichtigkeit je Gruppe von Spalten (Abfall von R² auf den Out-of-Bag-Zeilen, Mittel über die Bäume). Eine Gruppe mit mehreren Spalten wird mit **einer** gemeinsamen
    Umordnung der Zeilen gemischt (die Spalten der Gruppe bleiben zusammen)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    var_y = float(np.var(y))
    per_tree = []
    for tree, Xo, yo in _oob_trees(forest, X, y, min_oob):
        base = _mse(tree, Xo, yo)
        deltas = []
        for g in groups:
            perm = rng.permutation(len(yo))
            Xp = Xo.copy()
            for j in g:
                Xp[:, j] = Xo[perm, j]
            deltas.append((_mse(tree, Xp, yo) - base) / var_y)
        per_tree.append(deltas)
    if not per_tree:
        raise ValueError("kein Baum hat genug Out-of-Bag-Zeilen")
    return np.mean(per_tree, axis=0)


def quantile_classes(values, k: int) -> np.ndarray:
    """Klasse 0 bis k − 1 je Wert nach den Quantilen von `values` (gleich besetzte Klassen)."""
    values = np.asarray(values, dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, k + 1)[1:-1])
    return np.searchsorted(edges, values)


def oob_conditional_importance(forest: RF.Forest, X, y, column: int, given: int, seed: int = 0, classes: int = C.CLASSES, min_oob: int = C.MIN_OOB) -> float:
    """Bedingte Wichtigkeit von `column` gegeben `given`: die Spalte wird nur **innerhalb** der Quantilklassen der Spalte `given` umgeordnet, so bleibt die Abhängigkeit zwischen den beiden erhalten.
    Die Klassen stammen aus allen Zeilen, die Umordnung läuft über die Out-of-Bag-Zeilen jedes Baums."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    var_y = float(np.var(y))
    cls_all = quantile_classes(X[:, given], classes)
    deltas = []
    for b, tree in enumerate(forest.trees):
        oob = np.nonzero(~forest.in_bag[b])[0]
        if len(oob) < min_oob:
            continue
        Xo, yo, co = X[oob], y[oob], cls_all[oob]
        base = _mse(tree, Xo, yo)
        Xp = Xo.copy()
        for c in np.unique(co):
            idx = np.nonzero(co == c)[0]
            Xp[idx, column] = Xo[rng.permutation(idx), column]
        deltas.append((_mse(tree, Xp, yo) - base) / var_y)
    if not deltas:
        raise ValueError("kein Baum hat genug Out-of-Bag-Zeilen")
    return float(np.mean(deltas))


def permutation_importance(predict, X, y, groups, repeats: int = 20, seed: int = 0) -> np.ndarray:
    """Wichtigkeit je Gruppe für ein beliebiges Modell `predict(X)` auf allen Zeilen: Anstieg des mittleren Fehlerquadrats beim Mischen (gemeinsame Umordnung je Gruppe), geteilt durch Var(y),
    Mittel über `repeats` Umordnungen."""
    if repeats < 1:
        raise ValueError("repeats muss mindestens 1 sein")
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    base = float(np.mean((y - predict(X)) ** 2))
    var_y = float(np.var(y))
    out = np.zeros(len(groups))
    for _ in range(repeats):
        for i, g in enumerate(groups):
            perm = rng.permutation(len(y))
            Xp = X.copy()
            for j in g:
                Xp[:, j] = X[perm, j]
            out[i] += (float(np.mean((y - predict(Xp)) ** 2)) - base) / var_y
    return out / repeats


def share(a: float, b: float) -> float:
    """Anteil von a an a + b."""
    return a / (a + b)


# --- Lineares Modell -----------------------------------------------------------------------------------------------------------------------

def linear_importance_exact(b, cov, sigma_e: float, group) -> float:
    """Wichtigkeit einer Gruppe im linearen Modell mit **bekannten** Koeffizienten b: Das Mischen ersetzt die Gruppe durch eine unabhängige Kopie, der Fehler steigt um 2 b_Gᵀ Σ_GG b_G
    (der Rest ist unverändert), geteilt durch Var(y) = bᵀ Σ b + σ²."""
    b, cov = np.asarray(b, dtype=float), np.asarray(cov, dtype=float)
    g = list(group)
    var_y = float(b @ cov @ b) + sigma_e ** 2
    return float(2.0 * b[g] @ cov[np.ix_(g, g)] @ b[g]) / var_y


def fit_ols(X, y) -> np.ndarray:
    """Kleinste Quadrate mit Achsenabschnitt: Koeffizienten (Achsenabschnitt zuerst)."""
    A = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
    return beta


def ols_predict(beta: np.ndarray):
    return lambda X: beta[0] + np.asarray(X, dtype=float) @ beta[1:]


# --- Extrapolation ----------------------------------------------------------------------------------------------------------------------

def shuffle_column(X, column: int, seed: int) -> np.ndarray:
    """Die Spalte `column` in zufälliger Umordnung der Zeilen (dieselbe Umordnung für gleichen Seed)."""
    X = np.asarray(X, dtype=float)
    return X[np.random.default_rng(seed).permutation(len(X)), column]


def outside_share(X, column: int, partner: int, rho: float, seed: int, k: float = C.K_SIGMA) -> float:
    """Anteil der Zeilen, die nach dem Mischen von `column` außerhalb des k-Sigma-Bereichs der gemeinsamen Verteilung der beiden Zwillinge liegen
    (Mahalanobis-Abstand mit der wahren Korrelation rho)."""
    a = shuffle_column(X, column, seed)
    b = np.asarray(X, dtype=float)[:, partner]
    q = (a * a - 2.0 * rho * a * b + b * b) / (1.0 - rho * rho)
    return float(np.mean(q > k * k))


def outside_share_exact(rho: float, k: float = C.K_SIGMA) -> float:
    """Exakter Anteil nach dem Mischen: Q = U²/(1 − rho) + V²/(1 + rho) mit unabhängigen Standardnormalen U, V (Mahalanobis-Abstand zweier unabhängig gemischter Zwillinge);
    gesucht P(Q > k²). Innen ∫ φ(v) erfc(√((k² − V²/(1+rho))(1 − rho)/2)) dv über |v| < √(k²(1 + rho)), außen 1; Substitution v = v_end · sin θ macht den Rand glatt."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho muss in [0, 1) liegen")
    c = k * k
    v_end = math.sqrt(c * (1.0 + rho))
    nodes, weights = np.polynomial.legendre.leggauss(64)
    theta = 0.25 * math.pi * (nodes + 1.0)                      # [0, π/2]
    v = v_end * np.sin(theta)
    t = np.maximum(c - v * v / (1.0 + rho), 0.0)
    arg = np.sqrt(t * (1.0 - rho) / 2.0)
    inner = np.asarray(_erfc(arg), dtype=float)
    dens = np.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)
    half = 0.25 * math.pi * float(np.sum(weights * dens * inner * v_end * np.cos(theta)))
    inside_not_outside = 2.0 * half                              # P(|V| < v_end und Q > c)
    outside_v = math.erfc(v_end / math.sqrt(2.0))                # P(|V| ≥ v_end): dort ist Q > c sicher
    return inside_not_outside + outside_v


# --- Gruppen aus der Spearman-Korrelation ---------------------------------------------------------------------------------------------------

def groups_by_spearman(X, threshold: float) -> list[list[int]]:
    """Spalten, deren |Spearman| mit einer anderen mindestens `threshold` ist, kommen in dieselbe Gruppe (Vereinigung über Paare); jede Spalte steht in genau einer Gruppe, sortiert."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    parent = list(range(d))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(d):
        for j in range(i + 1, d):
            try:
                rank_corr = spearman_r(X[:, i], X[:, j])
            except ValueError:                                  # eine konstante Spalte hat keine Korrelation und bleibt allein
                continue
            if abs(rank_corr) >= threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for j in range(d):
        groups.setdefault(find(j), []).append(j)
    return sorted(groups.values())
