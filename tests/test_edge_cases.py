"""Randfälle (vor dem ersten Mutationstest): Reglergrenzen, ρ an den Enden, ohne Rauschmerkmale, kleinste Wälder, leere Gruppen."""

import numpy as np
import pytest

import kmk_algorithm as K
import kmk_constants as C
import kmk_evaluation as ev
import prs_scenario as S


@pytest.mark.parametrize("world", list(C.WORLDS))
@pytest.mark.parametrize("rho", [C.RHOS[0], C.RHOS[-1]])
def test_analyse_runs_at_both_ends_of_rho_in_both_worlds(world, rho):
    a = ev.analyse(ev.Settings(world, rho, C.N_MIN, C.TREES_MIN, 0, 0.9, 3))
    assert a.X.shape == (C.N_MIN, 2) and np.all(np.isfinite(a.singles)) and np.isfinite(a.group) and np.isfinite(a.cond_a) and 0.0 <= a.outside <= 1.0


def test_analyse_at_the_largest_forest():
    a = ev.analyse(ev.Settings("beide", 0.99, C.N_MAX, C.TREES_MAX, C.NOISE_MAX, 0.99, 1))
    assert a.X.shape == (C.N_MAX, 2 + C.NOISE_MAX) and len(a.singles) == 2 + C.NOISE_MAX and all(len(g) >= 1 for g in a.groups_found)


def test_oob_importance_raises_when_no_tree_has_enough_out_of_bag_rows():
    X, y = K.draw_twins(200, 0.5, "beide", 1, S.VRng(2))
    f = K.fit_forest(X, y, 5, 1)
    with pytest.raises(ValueError):
        K.oob_importance(f, X, y, [[0]], min_oob=10_000)
    with pytest.raises(ValueError):
        K.oob_conditional_importance(f, X, y, 0, 1, min_oob=10_000)


def test_oob_importance_of_an_empty_group_list_is_an_empty_array():
    X, y = K.draw_twins(200, 0.5, "beide", 0, S.VRng(2))
    f = K.fit_forest(X, y, 5, 1)
    assert K.oob_importance(f, X, y, [], seed=1).shape == (0,)


def test_share_is_defined_by_the_ratio_even_for_negative_values():
    assert K.share(-1.0, 3.0) == -0.5 and K.share(0.0, 2.0) == 0.0


def test_groups_by_spearman_with_a_single_column_or_constant_columns():
    assert K.groups_by_spearman(np.arange(10.0)[:, None], 0.9) == [[0]]
    X = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
    assert K.groups_by_spearman(X, 0.5) == [[0], [1]]                      # eine konstante Spalte hat keine Korrelation (0), bleibt allein


def test_conditional_importance_with_more_classes_than_rows_still_runs():
    X, y = K.draw_twins(200, 0.9, "beide", 0, S.VRng(5))
    f = K.fit_forest(X, y, 5, 2)
    assert np.isfinite(K.oob_conditional_importance(f, X, y, 0, 1, seed=1, classes=150))


def test_the_outside_share_stays_between_zero_and_one_and_grows_with_rho():
    vals = [K.outside_share_exact(r) for r in C.RHOS]
    assert vals == sorted(vals) and 0.0 < vals[0] and vals[-1] < 1.0
    assert K.outside_share_exact(0.5, 0.5) > K.outside_share_exact(0.5, 3.0)


def test_permutation_importance_needs_at_least_one_repeat():
    X = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError):
        K.permutation_importance(lambda M: M[:, 0], X, X[:, 0], [[0]], repeats=0)
    assert K.permutation_importance(lambda M: M[:, 0], X, X[:, 0], [[0]], repeats=1)[0] >= 0.0


def test_linear_importance_exact_of_the_empty_group_is_zero():
    b, cov, s = K.twin_model(0.5, "beide", 0)
    assert K.linear_importance_exact(b, cov, s, []) == 0.0


def test_a_tree_with_exactly_min_oob_out_of_bag_rows_still_counts():
    """Die Grenze ist inklusive: ein Baum mit genau `min_oob` Out-of-Bag-Zeilen zählt (sonst bräche die Rechnung hier ab, weil nur Bäume mit mehr Zeilen zählen würden)."""
    X, y = K.draw_twins(200, 0.5, "beide", 1, S.VRng(2))
    f = K.fit_forest(X, y, 3, 1)
    top = max(int((~f.in_bag[b]).sum()) for b in range(3))
    assert np.isfinite(K.oob_importance(f, X, y, [[0]], min_oob=top)[0]) and np.isfinite(K.oob_conditional_importance(f, X, y, 0, 1, min_oob=top))
    with pytest.raises(ValueError):
        K.oob_importance(f, X, y, [[0]], min_oob=top + 1)
    with pytest.raises(ValueError):
        K.oob_conditional_importance(f, X, y, 0, 1, min_oob=top + 1)


def test_the_spearman_threshold_is_inclusive():
    """Ein Paar, dessen |Spearman| genau der Schwelle gleicht, gehört in eine Gruppe."""
    from prs_algorithm import spearman_r
    rng = np.random.default_rng(4)
    a = rng.normal(size=50)
    b = a + 0.7 * rng.normal(size=50)
    exact = abs(spearman_r(a, b))
    assert K.groups_by_spearman(np.column_stack([a, b]), exact) == [[0, 1]]
    assert K.groups_by_spearman(np.column_stack([a, b]), exact + 1e-9) == [[0], [1]]


def test_outside_share_counts_only_rows_strictly_beyond_the_radius():
    """ρ = 0, A = 3 und B = 0: der Abstand ist genau 3 Sigma, die Zeile liegt auf dem Rand, nicht außerhalb."""
    X = np.array([[3.0, 0.0], [0.0, 0.0]])
    for seed in range(5):
        assert K.outside_share(X, 0, 1, 0.0, seed=seed) == 0.0
    assert K.outside_share(np.array([[3.1, 0.0], [0.0, 0.0]]), 0, 1, 0.0, seed=1) == 0.5
