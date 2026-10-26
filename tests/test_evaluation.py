"""Auswertung: Einzelwald, Zellen des Rasters (klein gerechnet), Zugriff auf das Raster."""

import numpy as np
import pytest

import kmk_algorithm as K
import kmk_constants as C
import kmk_evaluation as ev
import prs_scenario as S


def test_settings_equality_and_hash_use_the_values():
    assert ev.Settings() == ev.Settings() and hash(ev.Settings()) == hash(ev.Settings())
    assert ev.Settings(rho=0.9) != ev.Settings() and ev.Settings() != "x"
    assert len({ev.Settings(), ev.Settings(), ev.Settings(seed=1)}) == 2


def test_analyse_is_deterministic_and_consistent_with_the_kernel():
    s = ev.Settings("beide", 0.9, 250, 8, 2, 0.9, 7)
    a, b = ev.analyse(s), ev.analyse(s)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.singles, b.singles) and a.group == b.group and a.cond_a == b.cond_a
    assert a.X.shape == (250, 4) and len(a.singles) == 4 and a.shuffled_a.shape == (250,)
    data_seed = S.substream_seeds(7, 3)[0]
    X, y = K.draw_twins(250, 0.9, "beide", 2, S.VRng(data_seed))
    assert np.array_equal(a.X, X) and np.array_equal(a.y, y)
    assert a.outside_exact == K.outside_share_exact(0.9) and 0 <= a.outside <= 1
    assert a.exact["a"] == pytest.approx(2 / (2.25 + 1.8)) and a.exact["group"] == pytest.approx(4 * 1.9 / 4.05)


def test_analyse_groups_follow_the_threshold():
    strong = ev.analyse(ev.Settings("beide", 0.99, 300, 8, 1, 0.9, 3))
    weak = ev.analyse(ev.Settings("beide", 0.0, 300, 8, 1, 0.9, 3))
    assert strong.groups_found[0] == [0, 1] and weak.groups_found == [[0], [1], [2]]
    assert ev.analyse(ev.Settings("beide", 0.99, 300, 8, 1, 0.99, 3)).groups_found[0] in ([0, 1], [0])        # bei Schwelle 0,99 gerade an der Grenze


def test_analyse_without_noise_features():
    a = ev.analyse(ev.Settings("nur_a", 0.5, 200, 8, 0, 0.9, 1))
    assert a.X.shape == (200, 2) and len(a.singles) == 2


def test_cell_seed_differs_for_every_input():
    base = ev.cell_seed("twin", "beide", 0.5, 101)
    others = {ev.cell_seed("linear", "beide", 0.5, 101), ev.cell_seed("twin", "nur_a", 0.5, 101), ev.cell_seed("twin", "beide", 0.9, 101), ev.cell_seed("twin", "beide", 0.5, 102), ev.cell_seed("twin", "beide", 0.5, 101, extra=1)}
    assert base not in others and len(others) == 5


def test_twin_cell_small_run():
    cell = ev.twin_cell("beide", 0.9, 2, (7,), 250, 8, 2)
    assert cell["runs"] == 2 and cell["sum"] == pytest.approx(cell["a"] + cell["b"]) and cell["exact_group"] == pytest.approx(4 * 1.9 / 4.05)
    assert 0.0 <= cell["share_mean"] <= 1.0 and 0.0 <= cell["extreme"] <= 1.0 and cell["found"] in (0.0, 0.5, 1.0)
    assert cell == ev.twin_cell("beide", 0.9, 2, (7,), 250, 8, 2)
    assert ev.twin_cell("beide", 0.0, 1, (7,), 200, 6, 0)["noise"] == 0.0                        # ohne Rauschmerkmale: Rauschwert 0


def test_linear_cell_small_run():
    cell = ev.linear_cell(0.5, 3, (7,), 400, 1, 10)
    assert cell["runs"] == 3 and cell["sum"] == pytest.approx(cell["a"] + cell["b"]) and 0.3 < cell["share_mean"] < 0.7


def test_outside_cell_is_close_to_the_exact_value():
    cell = ev.outside_cell(0.9, 20, 400)
    assert cell["exact"] == K.outside_share_exact(0.9) and abs(cell["measured"] - cell["exact"]) < 0.03


def test_trees_cell_small_run():
    cell = ev.trees_cell(6, 0.9, 2, 4, 200)
    assert cell["trees"] == 6 and cell["same_data_sd"] >= 0 and cell["new_data_sd"] >= 0
    assert cell == ev.trees_cell(6, 0.9, 2, 4, 200)


def test_find_matches_floats_and_returns_none_when_missing():
    rows = [{"rho": 0.5, "world": "beide"}, {"rho": 0.9, "world": "nur_a"}]
    assert ev.find(rows, rho=0.9, world="nur_a") == rows[1] and ev.find(rows, rho=0.7) is None and ev.find(rows, rho=0.5 + 1e-12) == rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None


def test_the_grid_constants_cover_the_cells():
    assert set(C.GRID_RHOS) <= set(C.RHOS) and C.GROUP_THRESHOLD >= C.THRESHOLD_MIN
