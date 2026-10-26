"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/importance_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 und festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft, sonst gilt eine Toleranz von
einem halben Tausendstel bzw. Hundertstel. Die Zeilen-Bauer (`sum_rows` usw.) dienen auch dazu, die README-Tabellen zu erzeugen."""

from pathlib import Path

import pytest

import kmk_algorithm as K
import kmk_constants as C
import kmk_evaluation as ev

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v):
    return f"{100 * v:.1f}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def _beide(grid):
    return {r["rho"]: r for r in grid["twin"] if r["world"] == "beide"}


def _nur_a(grid):
    return {r["rho"]: r for r in grid["twin"] if r["world"] == "nur_a"}


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def sum_rows(grid) -> list[str]:
    lin = {r["rho"]: r for r in grid["linear"]}
    return [f"| {_c(rho, 3).rstrip('0').rstrip(',') if rho not in (0.0,) else '0'} | {_c(r['a'], 3)} | {_c(r['b'], 3)} | {_c(r['noise'], 3)} | {_c(r['sum'], 3)} | {_c(r['group'], 3)} | "
            f"{_c(r['exact_a'] + r['exact_b'], 3)} | {_c(lin[rho]['sum'], 3)} |" for rho, r in sorted(_beide(grid).items())]


def spread_rows(grid) -> list[str]:
    lin = {r["rho"]: r for r in grid["linear"]}
    return [f"| {_c(rho, 3).rstrip('0').rstrip(',') if rho != 0.0 else '0'} | {_c(r['share_mean'], 3)} | {_c(r['share_sd'], 3)} | {_p(r['extreme'])} % | {_c(lin[rho]['share_sd'], 3)} |" for rho, r in sorted(_beide(grid).items())]


def only_a_rows(grid) -> list[str]:
    return [f"| {_c(rho, 3).rstrip('0').rstrip(',') if rho != 0.0 else '0'} | {_c(r['a'], 3)} | {_c(r['b'], 3)} | {_c(r['ratio_b_a'], 3)} |" for rho, r in sorted(_nur_a(grid).items())]


def remedy_rows(grid) -> list[str]:
    return [f"| {_c(rho, 3).rstrip('0').rstrip(',') if rho != 0.0 else '0'} | {_c(r['a'], 3)} | {_c(r['b'], 3)} | {_c(r['group'], 3)} | {_c(r['group'] / r['sum'], 2)} | {_c(r['cond_a'], 3)} | "
            f"{_c(r['cond_b'], 3)} | {_p(r['found'])} % |" for rho, r in sorted(_beide(grid).items())]


def outside_rows(grid) -> list[str]:
    return [f"| {_c(r['rho'], 3).rstrip('0').rstrip(',') if r['rho'] != 0.0 else '0'} | {_p(r['measured'])} % | {_p(r['exact'])} % |" for r in sorted(grid["outside"], key=lambda r: r["rho"])]


def trees_rows(grid) -> list[str]:
    return [f"| {r['trees']} | {_c(r['same_data_sd'], 3)} | {_c(r['new_data_sd'], 3)} |" for r in sorted(grid["trees"], key=lambda r: r["trees"])]


# --- Tabellen der README --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [sum_rows, spread_rows, only_a_rows, remedy_rows, outside_rows, trees_rows])
def test_claim_the_tables_are_the_grid_in_the_apps_rounding(grid, builder):
    """README: die sechs Tabellen der Befunde (je fünf, fünf, fünf, fünf, fünf und drei Zeilen)."""
    rows = builder(grid)
    assert rows
    for row in rows:
        assert row in README, row


# --- Aussagen -----------------------------------------------------------------------------------------------------------------------------

def test_claim_the_sum_falls_and_the_group_stays(grid):
    """README: Summe 1,67 (ρ = 0) → 1,23 (ρ = 0,5); 1,22 / 1,42 / 1,46 bei 0,9 / 0,99 / 0,999; Gruppe 1,77 bis 1,87 ab ρ = 0,5 und 1,67 bei ρ = 0."""
    c = _beide(grid)
    assert [c[r]["sum"] for r in C.GRID_RHOS] == pytest.approx([1.67, 1.23, 1.22, 1.42, 1.46], abs=0.006)
    assert [c[r]["group"] for r in C.GRID_RHOS] == pytest.approx([1.67, 1.77, 1.84, 1.87, 1.86], abs=0.006)
    assert max(abs(r["noise"]) for r in grid["twin"]) < 0.005


def test_claim_the_exact_linear_values(grid):
    """README: im linearen Modell mit bekannten Koeffizienten ist die Summe 4/(2,25 + 2ρ): 1,78 / 1,23 / 0,99 / 0,95 / 0,94; die Gruppe 4(1 + ρ)/(2,25 + 2ρ): 1,78 → 1,88; angepasst: 1,79 / 1,23 / 0,99 / 0,97 / 1,14."""
    c = _beide(grid)
    assert [c[r]["exact_a"] + c[r]["exact_b"] for r in C.GRID_RHOS] == pytest.approx([1.78, 1.23, 0.99, 0.95, 0.94], abs=0.006)
    assert [4 / (2.25 + 2 * r) for r in C.GRID_RHOS] == pytest.approx([1.78, 1.23, 0.99, 0.95, 0.94], abs=0.006)
    assert c[0.0]["exact_group"] == pytest.approx(1.78, abs=0.006) and c[0.999]["exact_group"] == pytest.approx(1.88, abs=0.006)
    assert [r["sum"] for r in sorted(grid["linear"], key=lambda r: r["rho"])] == pytest.approx([1.79, 1.23, 0.99, 0.97, 1.14], abs=0.006)


def test_claim_the_split_jumps(grid):
    """README: Mittel der Aufteilung 0,48 bis 0,52; Streuung 0,032 / 0,040 / 0,143 / 0,181 / 0,135; Läufe mit mehr als 70 : 30: 0 / 0 / 17 / 33 / 20 %; angepasstes lineares Modell 0,024 / 0,023 / 0,048 / 0,140 / 0,302."""
    c = _beide(grid)
    assert all(0.48 <= c[r]["share_mean"] <= 0.52 for r in C.GRID_RHOS)
    assert [c[r]["share_sd"] for r in C.GRID_RHOS] == pytest.approx([0.032, 0.040, 0.143, 0.181, 0.135], abs=0.0006)
    assert [c[r]["extreme"] for r in C.GRID_RHOS] == pytest.approx([0.0, 0.0, 0.17, 0.33, 0.20], abs=0.0006)
    assert [r["share_sd"] for r in sorted(grid["linear"], key=lambda r: r["rho"])] == pytest.approx([0.024, 0.023, 0.048, 0.140, 0.302], abs=0.0006)


def test_claim_more_trees_do_not_calm_the_split(grid):
    """README: gleiche Daten, neuer Wald: 0,091 / 0,050 / 0,028 bei 10 / 30 / 100 Bäumen; neue Daten: 0,200 / 0,173 / 0,163."""
    rows = sorted(grid["trees"], key=lambda r: r["trees"])
    assert [r["same_data_sd"] for r in rows] == pytest.approx([0.091, 0.050, 0.028], abs=0.0006)
    assert [r["new_data_sd"] for r in rows] == pytest.approx([0.200, 0.173, 0.163], abs=0.0006)


def test_claim_a_copy_without_information(grid):
    """README: nur A trägt bei: B höchstens 0,002 bis ρ = 0,9, 0,080 bei 0,99 und 0,504 bei 0,999; A fällt von 1,87 auf 0,99."""
    c = _nur_a(grid)
    assert max(c[r]["b"] for r in (0.0, 0.5, 0.9)) == pytest.approx(0.002, abs=0.0006) and c[0.99]["b"] == pytest.approx(0.080, abs=0.0006) and c[0.999]["b"] == pytest.approx(0.504, abs=0.0006)
    assert (c[0.0]["a"], c[0.999]["a"]) == pytest.approx((1.87, 0.99), abs=0.006) and c[0.999]["sum"] == pytest.approx(1.50, abs=0.006)
    assert [c[r]["a"] for r in (0.0, 0.5, 0.9)] == pytest.approx([1.87, 1.86, 1.84], abs=0.006)


def test_claim_the_three_answers(grid):
    """README: bei ρ = 0,99: A einzeln 0,70, Gruppe 1,87, bedingt 0,04; bedingt auf B bei ρ = 0 / 0,5 / 0,9: 0,80 / 0,44 / 0,11; die Spearman-Schwelle 0,9 findet die Gruppe bei ρ = 0,9 in 17 %, ab ρ = 0,99 in 100 %."""
    c = _beide(grid)
    assert (c[0.99]["a"], c[0.99]["group"], c[0.99]["cond_a"]) == pytest.approx((0.70, 1.87, 0.04), abs=0.006)
    assert [c[r]["cond_a"] for r in (0.0, 0.5, 0.9)] == pytest.approx([0.80, 0.44, 0.11], abs=0.006)
    assert [c[r]["found"] for r in C.GRID_RHOS] == pytest.approx([0.0, 0.0, 0.17, 1.0, 1.0], abs=0.0006)
    assert (c[0.9]["group"] / c[0.9]["sum"], c[0.0]["group"] / c[0.0]["sum"]) == pytest.approx((1.50, 1.00), abs=0.006)


def test_claim_the_outside_statements(grid):
    """README: exakt 1,1 / 4,4 / 35,8 / 77,1 / 92,7 % bei ρ = 0 / 0,5 / 0,9 / 0,99 / 0,999; bei ρ = 0 e^(−4,5) = 1,11 %."""
    rows = sorted(grid["outside"], key=lambda r: r["rho"])
    assert [100 * r["exact"] for r in rows] == pytest.approx([1.1, 4.4, 35.8, 77.1, 92.7], abs=0.06)
    assert K.outside_share_exact(0.0) == pytest.approx(0.0111, abs=0.00006) and max(abs(r["measured"] - r["exact"]) for r in rows) < 0.005


def test_claim_hand_values():
    """README: Modell = 2 A, y = 2 A: Wichtigkeit 2 (Erwartung); zwei identische Eingaben einzeln 0,5, gemeinsam 2,0; bei ρ = 0 Anteil außerhalb e^(−4,5); Spearman findet {A, B} und {C} bei drei Spalten."""
    import numpy as np
    A = np.array([1.0, 2.0, 3.0, 4.0])
    X, y = np.column_stack([A, A, np.full(4, 5.0)]), 2.0 * A
    assert K.permutation_importance(lambda M: 2.0 * M[:, 0], X, y, [[0]], repeats=3000, seed=2)[0] == pytest.approx(2.0, abs=0.1)
    single = K.permutation_importance(lambda M: M[:, 0] + M[:, 1], X, y, [[0]], repeats=3000, seed=4)[0]
    group = K.permutation_importance(lambda M: M[:, 0] + M[:, 1], X, y, [[0, 1]], repeats=3000, seed=5)[0]
    assert (single, group) == pytest.approx((0.5, 2.0), abs=0.1)


def test_claim_the_grid_time(grid):
    """README: das Raster dauert 238 Sekunden auf 10 Prozessen (Wert aus der Datei; ändert sich beim Neuerzeugen)."""
    assert grid["meta"]["seconds"] == 238


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

def test_claim_the_preset_help_texts_quote_grid_numbers(grid):
    h, c = C.PRESET_HELP, _beide(grid)
    std = h["Standardfall (Voreinstellung)"]
    assert f"die Summe der Einzelwichtigkeiten {_c(c[0.5]['sum'], 2)} (bei ρ = 0: {_c(c[0.0]['sum'], 2)})" in std and f"die Gruppe aus A und B {_c(c[0.5]['group'], 2)}" in std and f"je Zwilling {_c(c[0.5]['cond_a'], 2)}" in std
    spr = h["Dieselbe Information, springende Aufteilung"]
    assert f"bleibt der mittlere Anteil von A bei {_p(c[0.99]['share_mean'])} %" in spr and f"streut mit {_c(c[0.99]['share_sd'], 3)}" in spr and f"in etwa einem Drittel der Läufe ({_p(c[0.99]['extreme'])} % bei 100 Wäldern)" in spr
    na = _nur_a(grid)
    only = h["Nur A trägt bei"]
    assert f"bekommt B {_c(na[0.999]['b'], 2)} und A nur noch {_c(na[0.999]['a'], 2)} (bei ρ = 0: A {_c(na[0.0]['a'], 2)}, B {_c(na[0.0]['b'], 2)})" in only
    assert f"höchstens {_c(max(na[r]['b'] for r in (0.0, 0.5, 0.9)), 3)}" in only
    none = h["Ohne Zwilling"]
    assert f"A {_c(c[0.0]['a'], 2)}, B {_c(c[0.0]['b'], 2)}, die Summe {_c(c[0.0]['sum'], 2)} und die Gruppe {_c(c[0.0]['group'], 2)}" in none and f"bedingt {_c(c[0.0]['cond_a'], 2)} und {_c(c[0.0]['cond_b'], 2)}" in none
    grp = h["Gruppe bilden"]
    assert f"die Gruppe {_c(c[0.9]['group'], 2)}, die Summe der Einzelwerte nur {_c(c[0.9]['sum'], 2)}" in grp and f"nur in {_p(c[0.9]['found'])} % der Läufe" in grp
    out = h["Mischen erzeugt Unmögliches"]
    assert f"exakt {_p(K.outside_share_exact(0.99))} %" in out and f"unabhängigen Merkmalen {_p(K.outside_share_exact(0.0))} %" in out
