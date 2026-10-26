"""Das vorgerechnete Raster (data/importance_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import kmk_algorithm as K
import kmk_constants as C
import kmk_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/importance_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-7, abs=1e-9), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Wald-Kern oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    def keys(rows, *names):
        return sorted(tuple(r[n] for n in names) for r in rows)

    assert keys(grid["twin"], "world", "rho") == sorted((w, r) for w in C.WORLDS for r in C.GRID_RHOS)
    assert keys(grid["linear"], "rho") == sorted((r,) for r in C.GRID_RHOS) and keys(grid["outside"], "rho") == sorted((r,) for r in C.GRID_RHOS)
    assert keys(grid["trees"], "trees") == sorted((t,) for t in C.TREE_COUNTS)


def test_grid_meta_and_run_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(r["runs"] == len(C.GRID_SEEDS) * C.GRID_RUNS for r in grid["twin"] + grid["linear"]) and all(r["datasets"] == C.OUTSIDE_SETS for r in grid["outside"])
    assert all(r["datasets"] == C.TREE_DATASETS and r["forests"] == C.TREE_FORESTS for r in grid["trees"])


def test_the_exact_columns_are_the_linear_formula(grid):
    for r in grid["twin"]:
        b, cov, sigma = K.twin_model(r["rho"], r["world"], C.GRID_NOISE)
        assert (r["exact_a"], r["exact_b"], r["exact_group"]) == pytest.approx([K.linear_importance_exact(b, cov, sigma, g) for g in ([0], [1], [0, 1])], rel=1e-12)
    for r in grid["outside"]:
        assert r["exact"] == pytest.approx(K.outside_share_exact(r["rho"]), rel=1e-12)


def test_sums_and_ratios_are_consistent(grid):
    for r in grid["twin"]:
        assert r["sum"] == pytest.approx(r["a"] + r["b"], abs=1e-12) and 0.0 <= r["found"] <= 1.0 and 0.0 <= r["extreme"] <= 1.0
        assert abs(r["noise"]) < 0.01                                          # Rauschmerkmale tragen nichts bei


def test_the_sum_falls_while_the_group_stays_up(grid):
    cells = {r["rho"]: r for r in grid["twin"] if r["world"] == "beide"}
    assert cells[0.0]["sum"] > cells[0.5]["sum"] + 0.3 and cells[0.0]["group"] == pytest.approx(cells[0.0]["sum"], abs=0.05)
    for rho in (0.5, 0.9, 0.99, 0.999):
        assert cells[rho]["group"] > cells[rho]["sum"] + 0.2 and cells[rho]["group"] > cells[0.0]["sum"]
    cond = [cells[r]["cond_a"] for r in C.GRID_RHOS]
    assert all(a > b - 0.005 for a, b in zip(cond, cond[1:])) and cond[0] > 0.7 and cond[-1] < 0.06                # bei ρ ≥ 0,99 ist der einzigartige Anteil praktisch null (Rauschen in der dritten Stelle)


def test_the_split_spreads_with_rho_but_stays_centered(grid):
    cells = {r["rho"]: r for r in grid["twin"] if r["world"] == "beide"}
    assert cells[0.0]["share_sd"] < 0.06 and cells[0.9]["share_sd"] > 0.1 and cells[0.99]["share_sd"] > 0.1
    assert all(abs(c["share_mean"] - 0.5) < 0.05 for c in cells.values())
    assert cells[0.0]["extreme"] == 0.0 and cells[0.99]["extreme"] > 0.15


def test_a_copy_without_information_stays_unimportant_until_rho_is_very_high(grid):
    cells = {r["rho"]: r for r in grid["twin"] if r["world"] == "nur_a"}
    for rho in (0.0, 0.5, 0.9):
        assert cells[rho]["b"] < 0.01 and cells[rho]["a"] > 1.7
    assert cells[0.99]["b"] > 0.03 and cells[0.999]["b"] > 0.3 and cells[0.999]["a"] < cells[0.0]["a"] - 0.5


def test_the_group_is_found_only_when_the_correlation_is_high_enough(grid):
    cells = {r["rho"]: r for r in grid["twin"] if r["world"] == "beide"}
    assert cells[0.0]["found"] == 0.0 and cells[0.5]["found"] == 0.0 and cells[0.99]["found"] == 1.0 and cells[0.999]["found"] == 1.0 and 0.0 < cells[0.9]["found"] < 0.5


def test_the_linear_model_splits_exactly_in_expectation_but_spreads_when_fitted(grid):
    lin = {r["rho"]: r for r in grid["linear"]}
    twin = {r["rho"]: r for r in grid["twin"] if r["world"] == "beide"}
    for rho, r in lin.items():
        assert abs(r["share_mean"] - 0.5) < 0.08
    assert lin[0.0]["sum"] == pytest.approx(twin[0.0]["exact_a"] + twin[0.0]["exact_b"], abs=0.05) and lin[0.5]["sum"] == pytest.approx(twin[0.5]["exact_a"] + twin[0.5]["exact_b"], abs=0.05)
    assert lin[0.0]["share_sd"] < 0.05 < lin[0.99]["share_sd"] < lin[0.999]["share_sd"]


def test_counted_outside_share_matches_the_exact_value(grid):
    for r in grid["outside"]:
        assert r["measured"] == pytest.approx(r["exact"], abs=0.01)
    assert grid["outside"][0]["exact"] < 0.012 and max(r["exact"] for r in grid["outside"]) > 0.9


def test_more_trees_calm_the_split_only_for_the_same_data(grid):
    rows = sorted(grid["trees"], key=lambda r: r["trees"])
    assert [r["same_data_sd"] for r in rows] == sorted((r["same_data_sd"] for r in rows), reverse=True)
    assert max(r["new_data_sd"] for r in rows) - min(r["new_data_sd"] for r in rows) < 0.08 and min(r["new_data_sd"] for r in rows) > 0.1
