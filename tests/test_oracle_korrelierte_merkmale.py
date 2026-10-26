"""Orakel (ORACLE-PLAYBOOK): jedes Verfahren gegen einen anderen Rechenweg.

* Allgemeine Permutationswichtigkeit gegen `sklearn.inspection.permutation_importance` (gleiche Definition: Abfall von R²) und gegen die **exakte Formel des linearen Modells**.
* Out-of-Bag-Wichtigkeit des Waldes gegen eine Schleifenfassung (je Zeile, derselbe Zufallsstrom); Gruppenmischen gegen die Schleife mit gemeinsamer Umordnung.
* Anteil außerhalb des Datenbereichs: Quadratur mit `scipy.integrate.quad`, Zählung an einer großen Stichprobe.
* Spearman gegen `scipy.stats.spearmanr`, Gruppenbildung gegen die Zählung der Paare über der Schwelle.
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit."""

import math

import numpy as np
import pytest

import kmk_algorithm as K
import prs_scenario as S
import rf_tree as T

scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")
sk_linear = pytest.importorskip("sklearn.linear_model")
sk_inspection = pytest.importorskip("sklearn.inspection")


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_general_importance_matches_sklearn(rho):
    X, y = K.draw_twins(3000, rho, "beide", 2, S.VRng(21))
    model = sk_linear.LinearRegression().fit(X, y)
    ref = sk_inspection.permutation_importance(model, X, y, scoring="r2", n_repeats=60, random_state=0).importances_mean
    mine = K.permutation_importance(model.predict, X, y, [[0], [1], [2], [3]], repeats=60, seed=1)
    assert mine == pytest.approx(ref, abs=0.04)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("world", ["beide", "nur_a"])
def test_general_importance_matches_the_exact_linear_formula(rho, world):
    """Modell = wahre Koeffizienten: Wichtigkeit einer Spalte 2 b² Σ_jj / Var(y), einer Gruppe 2 bᵀ Σ_GG b / Var(y)."""
    b, cov, sigma = K.twin_model(rho, world, 2)
    X, y = K.draw_twins(20_000, rho, world, 2, S.VRng(31))
    groups = [[0], [1], [2], [0, 1]]
    mine = K.permutation_importance(lambda M: M @ b, X, y, groups, repeats=8, seed=2)
    exact = [K.linear_importance_exact(b, cov, sigma, g) for g in groups]
    assert mine == pytest.approx(exact, abs=0.03)


def _loop_importance(forest, X, y, groups, seed, min_oob=10):
    """Schleifenfassung der Out-of-Bag-Wichtigkeit: derselbe Zufallsstrom, aber der Fehler wird Zeile für Zeile mit Python-Schleifen gerechnet."""
    rng = np.random.default_rng(seed)
    var_y = float(np.var(y))
    rows = []
    for b, tree in enumerate(forest.trees):
        oob = [i for i in range(len(y)) if not forest.in_bag[b][i]]
        if len(oob) < min_oob:
            continue
        base = sum((y[i] - T.predict_value(tree, X[i:i + 1])[0]) ** 2 for i in oob) / len(oob)
        deltas = []
        for g in groups:
            perm = rng.permutation(len(oob))
            err = 0.0
            for pos, i in enumerate(oob):
                row = X[i].copy()
                for j in g:
                    row[j] = X[oob[perm[pos]], j]
                err += (y[i] - T.predict_value(tree, row[None, :])[0]) ** 2
            deltas.append((err / len(oob) - base) / var_y)
        rows.append(deltas)
    return np.mean(rows, axis=0)


def test_oob_importance_matches_the_loop_version_for_single_columns_and_a_group():
    X, y = K.draw_twins(120, 0.9, "beide", 1, S.VRng(41))
    f = K.fit_forest(X, y, 6, 3)
    groups = [[0], [1], [2], [0, 1]]
    assert K.oob_importance(f, X, y, groups, seed=5) == pytest.approx(_loop_importance(f, X, y, groups, 5), abs=1e-12)


def test_conditional_importance_matches_a_loop_version():
    X, y = K.draw_twins(120, 0.9, "beide", 0, S.VRng(43))
    f = K.fit_forest(X, y, 6, 4)
    mine = K.oob_conditional_importance(f, X, y, 0, 1, seed=7, classes=4)
    rng = np.random.default_rng(7)
    cls = K.quantile_classes(X[:, 1], 4)
    var_y = float(np.var(y))
    deltas = []
    for b, tree in enumerate(f.trees):
        oob = np.nonzero(~f.in_bag[b])[0]
        if len(oob) < 10:
            continue
        Xo, yo, co = X[oob].copy(), y[oob], cls[oob]
        base = float(np.mean((yo - T.predict_value(tree, Xo)) ** 2))
        for c in np.unique(co):
            idx = np.nonzero(co == c)[0]
            shuffled = Xo[rng.permutation(idx), 0]
            for pos, target in enumerate(idx):
                Xo[target, 0] = shuffled[pos]
        deltas.append((float(np.mean((yo - T.predict_value(tree, Xo)) ** 2)) - base) / var_y)
    assert mine == pytest.approx(float(np.mean(deltas)), abs=1e-12)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99, 0.999])
def test_outside_share_exact_matches_scipy_quad(rho):
    a, b = 1.0 / (1.0 - rho), 1.0 / (1.0 + rho)

    def integrand(v):
        t = 9.0 - b * v * v
        return scipy_stats.norm.pdf(v) * (scipy_stats.chi2.sf(t / a, 1) if t > 0 else 1.0)

    ref = scipy_integrate.quad(integrand, -12.0, 12.0, limit=300, epsabs=1e-13, points=[-math.sqrt(9 / b), math.sqrt(9 / b)])[0]
    assert K.outside_share_exact(rho) == pytest.approx(ref, abs=1e-8)


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99])
def test_outside_share_count_matches_the_exact_value(rho):
    rng = S.VRng(51)
    z1, z2 = rng.normal(200_000), rng.normal(200_000)
    X = np.column_stack([z1, rho * z1 + math.sqrt(1 - rho * rho) * z2])
    assert K.outside_share(X, 0, 1, rho, seed=3) == pytest.approx(K.outside_share_exact(rho), abs=0.004)


def test_spearman_copy_matches_scipy_and_the_groups_follow_the_pair_count():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 5))
    X[:, 1] = X[:, 0] + 0.2 * rng.normal(size=300)
    X[:, 4] = X[:, 3] + 0.3 * rng.normal(size=300)
    rho = scipy_stats.spearmanr(X).statistic
    for thr in (0.5, 0.8, 0.95):
        pairs = {(i, j) for i in range(5) for j in range(i + 1, 5) if abs(rho[i, j]) >= thr}
        groups = K.groups_by_spearman(X, thr)
        together = {(i, j) for g in groups for i in g for j in g if i < j}
        assert pairs <= together                                          # jedes Paar über der Schwelle liegt in einer Gruppe
    assert K.groups_by_spearman(X, 0.8) == [[0, 1], [2], [3, 4]]


def test_forest_importance_ranks_the_informative_column_first_and_noise_near_zero():
    X, y = K.draw_twins(600, 0.0, "nur_a", 2, S.VRng(61))
    f = K.fit_forest(X, y, 20, 1)
    imp = K.oob_importance(f, X, y, [[0], [1], [2], [3]], seed=2)
    assert imp[0] > 0.5 and abs(imp[1]) < 0.05 and abs(imp[2]) < 0.05 and abs(imp[3]) < 0.05
