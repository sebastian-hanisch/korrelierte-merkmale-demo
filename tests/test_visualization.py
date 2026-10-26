"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import kmk_constants as C
import kmk_evaluation as ev
import kmk_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings("beide", 0.9, 250, 8, 2, 0.9, 3))
    figs = [V.build_twin_scatter(a.X, a.shuffled_a, 0.9, mixed=False), V.build_twin_scatter(a.X, a.shuffled_a, 0.9, mixed=True), V.build_importance_bars(a), V.build_sum_lines(grid["twin"], grid["linear"]),
            V.build_spread(grid["twin"], grid["linear"]), V.build_trees_chart(grid["trees"]), V.build_only_a(grid["twin"]), V.build_outside(grid["outside"]), V.build_remedies(grid["twin"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_ellipse_points_satisfy_the_mahalanobis_equation():
    for rho in (0.0, 0.5, 0.9, 0.99):
        xs, ys = V.ellipse_points(rho)
        x, y = np.array(xs), np.array(ys)
        q = (x * x - 2 * rho * x * y + y * y) / (1 - rho * rho)
        assert q == pytest.approx(np.full_like(q, C.K_SIGMA ** 2), abs=1e-9)


def test_scatter_splits_inside_and_outside_after_mixing():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 3.0], [0.0, -3.0]])
    shuffled = np.array([4.0, 0.0, 0.0, 0.0])                                    # (4, 0) liegt bei rho = 0 außerhalb von 3 Sigma
    fig = V.build_twin_scatter(X, shuffled, 0.0, mixed=True)
    names = [t.name for t in fig.data]
    assert names == ["gemischt, im Datenbereich", "gemischt, außerhalb (gibt es nicht)", "3-Sigma-Bereich"]
    assert list(fig.data[1].x) == [4.0] and len(fig.data[0].x) == 3
    real = V.build_twin_scatter(X, shuffled, 0.0, mixed=False)
    assert [t.name for t in real.data] == ["echte Wochen", "3-Sigma-Bereich"] and list(real.data[0].x) == [0.0, 1.0, 0.0, 0.0]
    assert list(real.layout.xaxis.range) == [-V.AXIS_LIM, V.AXIS_LIM]


def test_importance_bars_have_one_bar_per_column_and_three_derived_ones():
    a = ev.analyse(ev.Settings("beide", 0.5, 250, 8, 2, 0.9, 3))
    fig = V.build_importance_bars(a)
    assert list(fig.data[0].x) == ["A einzeln", "B einzeln", "Rauschen 1", "Rauschen 2", "A + B als Gruppe", "A bedingt auf B", "B bedingt auf A"]
    assert list(fig.data[0].y)[:4] == pytest.approx(list(a.singles)) and fig.data[0].y[4] == pytest.approx(a.group)


def test_sum_chart_has_four_lines_in_the_order_of_rho(grid):
    fig = V.build_sum_lines(grid["twin"], grid["linear"])
    assert [t.name for t in fig.data] == ["Wald: Summe der Einzelwerte", "Wald: Gruppe A + B", "linear, bekannte Koeffizienten: Summe (exakt)", "linear, angepasst: Summe"]
    assert list(fig.data[0].x) == list(C.GRID_RHOS)
    first = ev.find(grid["twin"], world="beide", rho=0.0)
    assert fig.data[0].y[0] == pytest.approx(first["sum"]) and fig.data[2].y[0] == pytest.approx(first["exact_a"] + first["exact_b"])


def test_spread_and_trees_charts(grid):
    fig = V.build_spread(grid["twin"], grid["linear"])
    assert [t.name for t in fig.data] == ["Wald", "lineares Modell, angepasst"] and list(fig.data[0].x) == list(C.GRID_RHOS)
    trees = V.build_trees_chart(grid["trees"])
    assert list(trees.data[0].x) == [str(t) for t in sorted(C.TREE_COUNTS)] and trees.layout.barmode == "group"
    assert [t.name for t in trees.data] == ["gleiche Daten, neuer Wald", "neue Daten"]


def test_only_a_chart_uses_only_the_world_where_a_contributes(grid):
    fig = V.build_only_a(grid["twin"])
    cells = sorted((r for r in grid["twin"] if r["world"] == "nur_a"), key=lambda r: r["rho"])
    assert list(fig.data[1].y) == pytest.approx([r["b"] for r in cells]) and [t.name for t in fig.data] == ["A (trägt bei)", "B (verrauschte Kopie)"]


def test_outside_chart_shows_counted_bars_and_exact_diamonds(grid):
    fig = V.build_outside(grid["outside"])
    assert list(fig.data[0].x) == ["0,0", "0,5", "0,9", "0,99", "0,999"] and fig.data[1].y[3] == pytest.approx(100 * ev.find(grid["outside"], rho=0.99)["exact"]) and list(fig.layout.yaxis.range) == [0, 100]


def test_remedies_chart_groups_three_bars_per_rho(grid):
    fig = V.build_remedies(grid["twin"])
    assert [t.name for t in fig.data] == ["A einzeln", "A + B als Gruppe", "A bedingt auf B"] and fig.layout.barmode == "group" and len(fig.data[0].x) == len(C.GRID_RHOS)
