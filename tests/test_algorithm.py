"""Einheiten von Hand (DEMO-PLAYBOOK §1b): jede Funktion an einer Mini-Instanz, deren Ergebnis ohne Rechner folgt."""

import math

import numpy as np
import pytest

import kmk_algorithm as K
import kmk_constants as C
import prs_scenario as S


# --- Welt und Daten ------------------------------------------------------------------------------------------------------------------------

def test_twin_model_worlds():
    b, cov, sigma = K.twin_model(0.5, "beide", 2)
    assert list(b) == [1.0, 1.0, 0.0, 0.0] and cov[0, 1] == cov[1, 0] == 0.5 and cov[0, 0] == cov[1, 1] == 1.0 and cov[2, 3] == 0.0 and sigma == 0.5
    b2, _c, _s = K.twin_model(0.9, "nur_a", 0)
    assert list(b2) == [2.0, 0.0] and len(K.twin_model(0.0, "beide", 4)[0]) == 6
    with pytest.raises(ValueError):
        K.twin_model(0.5, "beide_und_noch_mehr", 2)


def test_draw_twins_shape_determinism_and_correlation():
    X, y = K.draw_twins(3000, 0.9, "beide", 2, S.VRng(5))
    X2, y2 = K.draw_twins(3000, 0.9, "beide", 2, S.VRng(5))
    assert X.shape == (3000, 4) and y.shape == (3000,) and np.array_equal(X, X2) and np.array_equal(y, y2)
    c = np.corrcoef(X, rowvar=False)
    assert c[0, 1] == pytest.approx(0.9, abs=0.02) and abs(c[0, 2]) < 0.05 and abs(c[2, 3]) < 0.05
    assert float(np.var(y)) == pytest.approx(2 + 2 * 0.9 + 0.25, rel=0.06)                 # Var(A + B) + Var(Rauschen)


def test_draw_twins_world_nur_a_reads_only_a():
    X, y = K.draw_twins(4000, 0.5, "nur_a", 2, S.VRng(5))
    assert np.corrcoef(X[:, 0], y)[0, 1] > 0.97 and abs(np.corrcoef(X[:, 2], y)[0, 1]) < 0.05
    assert float(np.var(y)) == pytest.approx(4.25, rel=0.06)


def test_draw_twins_with_rho_zero_and_no_noise_features():
    X, _y = K.draw_twins(3000, 0.0, "beide", 0, S.VRng(7))
    assert X.shape == (3000, 2) and abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) < 0.05


def test_draw_twins_rejects_a_correlation_of_one():
    for rho in (1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            K.draw_twins(10, rho, "beide", 0, S.VRng(1))


def test_fit_forest_grows_the_requested_trees_with_out_of_bag_rows():
    X, y = K.draw_twins(200, 0.5, "beide", 2, S.VRng(3))
    f = K.fit_forest(X, y, 7, 1)
    assert len(f.trees) == 7 and f.in_bag.shape == (7, 200) and f.mtry == 4
    assert 0.3 < float((~f.in_bag).mean()) < 0.45                              # etwa 1/e der Zeilen liegen außerhalb des Bootstrap


# --- Wichtigkeit eines beliebigen Modells -----------------------------------------------------------------------------------------------------------

def test_a_model_that_ignores_a_column_gives_that_column_exactly_zero(mini_rows):
    X, y = mini_rows
    imp = K.permutation_importance(lambda M: 2.0 * M[:, 0], X, y, [[1], [2]], repeats=5, seed=1)
    assert list(imp) == [0.0, 0.0]


def test_a_constant_column_has_zero_importance_even_if_the_model_reads_it(mini_rows):
    X, y = mini_rows
    imp = K.permutation_importance(lambda M: 2.0 * M[:, 0] + 0.0 * M[:, 2], X, y, [[2]], repeats=5, seed=1)
    assert imp[0] == 0.0


def test_permuting_the_only_input_gives_two_in_expectation(mini_rows):
    """Modell = 2 A, y = 2 A: der Fehler nach dem Mischen ist im Mittel 4 · 2 Var(A) = 10, Var(y) = 5, also Wichtigkeit 2 (Mittel über alle Umordnungen)."""
    X, y = mini_rows
    imp = K.permutation_importance(lambda M: 2.0 * M[:, 0], X, y, [[0]], repeats=3000, seed=2)
    assert imp[0] == pytest.approx(2.0, abs=0.08)


def test_a_group_is_shuffled_with_one_common_permutation(mini_rows):
    X, y = mini_rows
    seen = []

    def predict(M):
        seen.append(bool(np.array_equal(M[:, 0], M[:, 1])))
        return M[:, 0] + M[:, 1]

    K.permutation_importance(predict, X, y, [[0, 1]], repeats=20, seed=3)
    assert all(seen)                                                          # A und B waren Kopien und bleiben es nach dem gemeinsamen Mischen
    seen.clear()
    K.permutation_importance(predict, X, y, [[0], [1]], repeats=20, seed=3)
    assert not all(seen)                                                      # einzeln gemischt, trennen sich die Kopien


def test_group_versus_single_importance_for_two_identical_inputs(mini_rows):
    """Modell = A + B mit B = A, y = 2 A, Var(y) = 5: einzeln gemischt ist der Fehler (A − A_π)², im Mittel 2 Var(A) = 2,5, also 0,5; gemeinsam 4 · 2,5 = 10, also 2,0."""
    X, y = mini_rows
    single = K.permutation_importance(lambda M: M[:, 0] + M[:, 1], X, y, [[0], [1]], repeats=3000, seed=4)
    group = K.permutation_importance(lambda M: M[:, 0] + M[:, 1], X, y, [[0, 1]], repeats=3000, seed=5)
    assert list(single) == pytest.approx([0.5, 0.5], abs=0.04) and group[0] == pytest.approx(2.0, abs=0.08)


def test_share_of_two_importances():
    assert K.share(1.0, 3.0) == 0.25 and K.share(2.0, 2.0) == 0.5


# --- Lineares Modell -----------------------------------------------------------------------------------------------------------------------

def test_linear_importance_exact_hand_values():
    """b = (1, 1), unabhängig, σ = 0,5: Var(y) = 2,25; einzeln 2/2,25 = 0,8889, beide zusammen 4/2,25 = 1,7778; bei ρ = 1 fällt die Gruppe auf 2 · 4/4,25 = 1,8824 (Var(y) = 4,25)."""
    b, cov, s = K.twin_model(0.0, "beide", 2)
    assert K.linear_importance_exact(b, cov, s, [0]) == pytest.approx(2 / 2.25) and K.linear_importance_exact(b, cov, s, [0, 1]) == pytest.approx(4 / 2.25)
    assert K.linear_importance_exact(b, cov, s, [2]) == 0.0
    b1, cov1, s1 = K.twin_model(0.0, "beide", 0)
    cov1[0, 1] = cov1[1, 0] = 1.0
    assert K.linear_importance_exact(b1, cov1, s1, [0, 1]) == pytest.approx(8 / 4.25)


def test_linear_single_importance_formula_for_any_rho():
    for rho in (0.0, 0.5, 0.9, 0.99):
        b, cov, s = K.twin_model(rho, "beide", 2)
        assert K.linear_importance_exact(b, cov, s, [0]) == pytest.approx(2 / (2.25 + 2 * rho))


def test_fit_ols_recovers_exact_coefficients():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
    beta = K.fit_ols(X, y)
    assert beta == pytest.approx([1.0, 2.0, -3.0], abs=1e-12) and K.ols_predict(beta)(X) == pytest.approx(y, abs=1e-12)


# --- Klassen und bedingtes Mischen ---------------------------------------------------------------------------------------------------------

def test_quantile_classes_split_into_equal_groups():
    assert list(K.quantile_classes(np.arange(10.0), 2)) == [0] * 5 + [1] * 5
    assert list(K.quantile_classes(np.arange(8.0), 4)) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert set(K.quantile_classes(np.random.default_rng(1).normal(size=1000), 10)) == set(range(10))


def test_conditional_importance_with_one_class_equals_the_ordinary_one():
    """Mit einer einzigen Klasse ist das bedingte Mischen das gewöhnliche Mischen der einen Spalte (gleiche Zahl bei gleicher Umordnung des Zufallsstroms)."""
    X, y = K.draw_twins(300, 0.9, "beide", 2, S.VRng(4))
    f = K.fit_forest(X, y, 8, 2)
    one = K.oob_conditional_importance(f, X, y, 0, 1, seed=6, classes=1)
    ordinary = K.oob_importance(f, X, y, [[0]], seed=6)[0]
    assert one == pytest.approx(ordinary, abs=1e-12)


# --- Extrapolation ----------------------------------------------------------------------------------------------------------------------

def test_outside_share_exact_for_independent_twins_is_the_chi_square_tail():
    """ρ = 0: Q = U² + V² ist χ² mit zwei Freiheitsgraden, P(Q > k²) = e^(−k²/2): k = 3: e^(−4,5) = 0,011109; k = 1: e^(−0,5); k = 2: e^(−2)."""
    for k in (1.0, 2.0, 3.0):
        assert K.outside_share_exact(0.0, k) == pytest.approx(math.exp(-k * k / 2.0), rel=1e-9)


def test_outside_share_exact_grows_to_one_with_rho():
    vals = [K.outside_share_exact(r) for r in (0.0, 0.5, 0.9, 0.99, 0.999)]
    assert vals == sorted(vals) and vals[-1] > 0.9 and K.outside_share_exact(0.99999) > 0.99
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            K.outside_share_exact(bad)


def test_outside_share_counts_rows_beyond_the_ellipse():
    """ρ = 0: A = (5, 0, 0, 0) gemischt, B = (0, 0, 0, 0): die Zeile mit A = 5 liegt außerhalb von 3 Sigma, wohin immer sie wandert: Anteil 1/4."""
    X = np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert K.outside_share(X, 0, 1, 0.0, seed=1) == 0.25
    assert K.outside_share(np.zeros((6, 2)), 0, 1, 0.0, seed=1) == 0.0


# --- Gruppen aus der Spearman-Korrelation ----------------------------------------------------------------------------------------------

def test_groups_by_spearman_on_a_hand_instance():
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    X = np.column_stack([x, np.exp(x), rng.normal(size=60)])                # Spalte 1 ist eine monotone Umformung von Spalte 0: Spearman 1
    assert K.groups_by_spearman(X, 0.9) == [[0, 1], [2]]
    assert K.groups_by_spearman(X, 1.01) == [[0], [1], [2]]
    assert K.groups_by_spearman(X, 0.0) == [[0, 1, 2]]


def test_groups_by_spearman_links_through_a_chain():
    """A–B und B–C über der Schwelle, A–C darunter: alle drei in einer Gruppe (Vereinigung über Paare)."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=400)
    b = a + 0.5 * rng.normal(size=400)
    c = b + 0.5 * rng.normal(size=400)
    from prs_algorithm import spearman_r
    assert abs(spearman_r(a, b)) >= 0.8 and abs(spearman_r(b, c)) >= 0.8 and abs(spearman_r(a, c)) < 0.8
    assert K.groups_by_spearman(np.column_stack([a, b, c]), 0.8) == [[0, 1, 2]]


def test_constants_are_consistent():
    assert all(0.0 <= r < 1.0 for r in C.RHOS) and C.DEFAULT_RHO in C.RHOS and set(C.WORLDS) == set(C.WORLD_SHORT)
