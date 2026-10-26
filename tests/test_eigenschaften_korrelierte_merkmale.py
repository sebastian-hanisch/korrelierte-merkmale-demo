"""Eigenschaften: Invarianzen der rangbasierten Bäume, Symmetrie der Zwillinge, Zusammenhänge zwischen Einzel-, Gruppen- und bedingter Wichtigkeit, Reproduzierbarkeit."""

import numpy as np
import pytest

import kmk_algorithm as K
import prs_scenario as S


def _importances(X, y, groups, seed=1, trees=12):
    return K.oob_importance(K.fit_forest(X, y, trees, seed), X, y, groups, seed=seed + 1)


def test_the_importances_do_not_change_under_affine_transformations_of_the_columns():
    """Verschieben und Strecken einer Spalte ändert weder die Bäume noch die Wichtigkeit (gleiche Seeds): die Schwellen der Bäume liegen in der Mitte zwischen zwei Werten und gehen mit."""
    X, y = K.draw_twins(300, 0.9, "beide", 2, S.VRng(3))
    Xt = X.copy()
    Xt[:, 0] = 5.0 * X[:, 0] + 2.0
    Xt[:, 1] = 0.25 * X[:, 1] - 7.0
    groups = [[0], [1], [2], [0, 1]]
    assert _importances(Xt, y, groups) == pytest.approx(_importances(X, y, groups), abs=1e-12)


def test_the_importances_change_only_slightly_under_a_monotone_nonlinear_transformation():
    """Die Bäume schneiden nur nach Rängen der Trainingszeilen; die Schwellen liegen aber in der Wertemitte, daher ändern sich die Vorhersagen für die Out-of-Bag-Zeilen leicht (hier um höchstens 0,01)."""
    X, y = K.draw_twins(300, 0.9, "beide", 2, S.VRng(3))
    Xt = X.copy()
    Xt[:, 0] = np.exp(X[:, 0])
    groups = [[0], [1], [2], [0, 1]]
    assert _importances(Xt, y, groups) == pytest.approx(_importances(X, y, groups), abs=0.01)


def test_the_same_seeds_give_the_same_numbers_and_different_seeds_do_not():
    X, y = K.draw_twins(300, 0.9, "beide", 2, S.VRng(3))
    a, b = _importances(X, y, [[0], [1]], seed=1), _importances(X, y, [[0], [1]], seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, _importances(X, y, [[0], [1]], seed=2))


def test_twins_are_exchangeable_in_the_mean():
    """In der Welt „beide“ sind A und B austauschbar: über viele Wälder haben sie dieselbe mittlere Wichtigkeit."""
    rng = S.VRng(9)
    diffs = []
    for r in range(30):
        X, y = K.draw_twins(300, 0.9, "beide", 0, rng)
        imp = _importances(X, y, [[0], [1]], seed=r)
        diffs.append(imp[0] - imp[1])
    assert abs(float(np.mean(diffs))) < 3.0 * float(np.std(diffs, ddof=1)) / np.sqrt(len(diffs))


def test_swapping_the_twin_columns_swaps_their_importances_in_the_mean():
    rng = S.VRng(11)
    first, second = [], []
    for r in range(15):
        X, y = K.draw_twins(300, 0.5, "nur_a", 1, rng)
        first.append(_importances(X, y, [[0], [1]], seed=r))
        second.append(_importances(X[:, [1, 0, 2]], y, [[0], [1]], seed=r))
    f, s = np.mean(first, axis=0), np.mean(second, axis=0)
    assert f[0] > 1.0 and s[1] > 1.0 and f[1] < 0.05 and s[0] < 0.05


def test_group_importance_is_larger_than_each_single_importance_in_the_mean():
    rng = S.VRng(13)
    rows = []
    for r in range(20):
        X, y = K.draw_twins(300, 0.9, "beide", 1, rng)
        rows.append(_importances(X, y, [[0], [1], [0, 1]], seed=r))
    m = np.mean(rows, axis=0)
    assert m[2] > m[0] and m[2] > m[1]


def test_conditional_importance_is_smaller_than_the_single_one_in_the_mean_for_twins():
    rng = S.VRng(17)
    single, cond = [], []
    for r in range(20):
        X, y = K.draw_twins(300, 0.9, "beide", 1, rng)
        f = K.fit_forest(X, y, 12, r)
        single.append(K.oob_importance(f, X, y, [[0]], seed=r)[0])
        cond.append(K.oob_conditional_importance(f, X, y, 0, 1, seed=r))
    assert np.mean(cond) < np.mean(single)


def test_noise_columns_have_importance_near_zero_for_every_rho():
    for rho in (0.0, 0.9, 0.999):
        X, y = K.draw_twins(400, rho, "beide", 3, S.VRng(21))
        assert np.all(np.abs(_importances(X, y, [[2], [3], [4]], trees=15)) < 0.03)


def test_reordering_the_rows_before_fitting_changes_the_importances_only_by_noise():
    X, y = K.draw_twins(500, 0.0, "beide", 1, S.VRng(23))
    order = np.random.default_rng(1).permutation(500)
    a = _importances(X, y, [[0], [1]], trees=25)
    b = _importances(X[order], y[order], [[0], [1]], trees=25)
    assert a == pytest.approx(b, abs=0.12)


def test_the_exact_linear_importance_of_independent_columns_adds_up():
    """Bei ρ = 0 ist die Gruppe die Summe der Einzelwerte (Σ_GG diagonal), bei ρ > 0 größer."""
    for rho, bigger in ((0.0, False), (0.5, True)):
        b, cov, s = K.twin_model(rho, "beide", 0)
        single = K.linear_importance_exact(b, cov, s, [0]) + K.linear_importance_exact(b, cov, s, [1])
        group = K.linear_importance_exact(b, cov, s, [0, 1])
        assert (group > single + 1e-9) is bigger
        assert bigger or group == pytest.approx(single)
