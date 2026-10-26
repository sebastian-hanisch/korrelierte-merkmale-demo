"""Verdikte (reine Funktionen in kmk_evaluation.py) an handgesetzten Werten: Art und jede Zahl im Text."""

import numpy as np

import kmk_constants as C
import kmk_evaluation as ev


def _analysis(world="beide", rho=0.5, singles=(0.61, 0.62, 0.0, 0.0), group=1.77, cond=(0.44, 0.45), found=((0, 1), (2,), (3,)), outside=(0.045, 0.0439), exact=(0.6, 0.6, 1.8), threshold=0.9):
    s = ev.Settings(world, rho, 600, 30, 2, threshold, 1)
    return ev.Analysis(settings=s, X=np.zeros((1, 4)), y=np.zeros(1), shuffled_a=np.zeros(1), singles=np.array(singles), group=group, cond_a=cond[0], cond_b=cond[1],
                       groups_found=[list(g) for g in found], outside=outside[0], outside_exact=outside[1], exact={"a": exact[0], "b": exact[1], "group": exact[2]})


def test_verdict_main_names_every_number_in_the_world_with_two_contributors():
    kind, text = ev.verdict_main(_analysis())
    assert kind == "info"
    for needle in ("A **0,61**, B **0,62**", "Rauschmerkmale höchstens 0,00", "**1,77**", "innerhalb von 10 Klassen", "A **0,44**, B **0,45**", "49,6 % : 50,4 %"):
        assert needle in text, needle
    assert "fast alles" not in text


def test_verdict_main_warns_when_one_twin_gets_nearly_everything():
    kind, text = ev.verdict_main(_analysis(singles=(1.0, 0.1, 0.0, 0.0)))
    assert kind == "warning" and "90,9 % : 9,1 %" in text and "Einer der beiden Zwillinge bekommt fast alles" in text
    assert ev.verdict_main(_analysis(singles=(0.3, 0.7, 0, 0)))[0] == "info" and ev.verdict_main(_analysis(singles=(0.29, 0.71, 0, 0)))[0] == "warning"        # Schwelle bei 30 : 70


def test_verdict_main_without_noise_features_names_no_noise_value():
    _, text = ev.verdict_main(_analysis(singles=(0.6, 0.6)))
    assert "Rauschmerkmale" not in text


def test_verdict_main_in_the_world_where_only_a_contributes():
    kind, text = ev.verdict_main(_analysis("nur_a", singles=(1.0, 0.2, 0.0, 0.0)))
    assert kind == "warning" and "Nur A trägt bei, B bekommt trotzdem 16,7 % der Summe" in text
    kind, text = ev.verdict_main(_analysis("nur_a", singles=(1.0, 0.1, 0.0, 0.0)))
    assert kind == "info" and "B trägt nichts bei" in text


def test_verdict_exact_reports_the_known_coefficient_values():
    kind, text = ev.verdict_exact(_analysis())
    assert kind == "info" and "A **0,60**, B **0,60**, Gruppe **1,80**" in text and "Summe der Einzelwerte 1,20" in text and "50 : 50" in text
    _, text = ev.verdict_exact(_analysis(exact=(0.7, 0.5, 1.8)))
    assert "58,3 % : 41,7 %" in text


def test_verdict_outside_warns_above_twenty_percent():
    kind, text = ev.verdict_outside(_analysis(outside=(0.77, 0.7709)))
    assert kind == "warning" and "**77,0 %**" in text and "exakt: 77,1 %" in text and "1,1 %" in text and "extrapoliert" in text
    kind, text = ev.verdict_outside(_analysis(outside=(0.045, 0.0439)))
    assert kind == "info" and "extrapoliert" not in text
    assert ev.verdict_outside(_analysis(outside=(0.2, 0.2)))[0] == "info" and ev.verdict_outside(_analysis(outside=(0.2, 0.2001)))[0] == "warning"


def test_verdict_groups_succeeds_only_when_a_and_b_are_in_one_group():
    kind, text = ev.verdict_groups(_analysis(rho=0.99, threshold=0.9))
    assert kind == "success" and "|Spearman| ≥ 0,90" in text and "A + B" in text and "**1,77**" in text
    kind, text = ev.verdict_groups(_analysis(found=((0,), (1,), (2,), (3,))))
    assert kind == "warning" and "keine** Gruppe aus A und B" in text and "1,77" in text
    kind, text = ev.verdict_groups(_analysis(found=((0, 2), (1,), (3,))))
    assert kind == "warning"                                                           # A mit einem Rauschmerkmal ist keine Zwillingsgruppe


def _twin_rows():
    def cell(world, rho, a, b, s, g, ca, cb, sd, ext, found):
        return {"world": world, "rho": rho, "runs": 100, "a": a, "b": b, "noise": 0.0, "sum": s, "group": g, "cond_a": ca, "cond_b": cb, "share_mean": 0.5, "share_sd": sd, "extreme": ext, "found": found,
                "ratio_b_a": b / a, "exact_a": 0.6, "exact_b": 0.6, "exact_group": 1.8}
    return [cell("beide", 0.0, 0.83, 0.84, 1.67, 1.67, 0.8, 0.8, 0.03, 0.0, 0.0), cell("beide", 0.5, 0.61, 0.62, 1.23, 1.77, 0.44, 0.44, 0.04, 0.0, 0.0),
            cell("beide", 0.99, 0.7, 0.72, 1.42, 1.87, 0.04, 0.04, 0.18, 0.33, 1.0),
            cell("nur_a", 0.0, 1.87, 0.0, 1.87, 1.87, 1.8, 0.0, 0.0, 1.0, 0.0), cell("nur_a", 0.9, 1.84, 0.002, 1.84, 1.86, 0.4, 0.0, 0.0, 1.0, 0.2), cell("nur_a", 0.999, 0.99, 0.5, 1.5, 1.87, 0.05, 0.02, 0.16, 0.5, 1.0)]


def _linear_rows():
    return [{"rho": 0.0, "share_sd": 0.02, "sum": 1.78}, {"rho": 0.5, "share_sd": 0.02, "sum": 1.23}, {"rho": 0.99, "share_sd": 0.14, "sum": 0.97}]


def test_verdict_sum_grid_reads_first_middle_and_last_rho():
    kind, text = ev.verdict_sum_grid(_twin_rows(), _linear_rows())
    assert kind == "warning"
    for needle in ("**1,67** bei ρ = 0", "**1,23** bei ρ = 0,5", "1,42 bei ρ = 0,99", "bleibt bei 1,77 bis 1,87 (ρ = 0: 1,67)", "bei ρ = 0,5 exakt 1,20"):
        assert needle in text, needle


def test_verdict_spread_warns_above_a_spread_of_a_tenth():
    kind, text = ev.verdict_spread(_twin_rows(), _linear_rows(), 0.99)
    assert kind == "warning" and "**0,180**" in text and "**33,0 %**" in text and "mehr als 70,0 %" in text and "0,140" in text
    assert ev.verdict_spread(_twin_rows(), _linear_rows(), 0.5)[0] == "info"


def test_verdict_trees_compares_same_data_and_new_data():
    rows = [{"trees": 100, "rho": 0.99, "same_data_sd": 0.023, "new_data_sd": 0.178}, {"trees": 10, "rho": 0.99, "same_data_sd": 0.082, "new_data_sd": 0.182}]
    kind, text = ev.verdict_trees(rows)
    assert kind == "warning"
    for needle in ("mit 10 Bäumen", "0,082", "mit 100 Bäumen nur noch mit 0,023", "0,182 und 0,178", "mehr Bäume beruhigen sie kaum"):
        assert needle in text, needle


def test_verdict_only_a_reads_the_calm_range_and_the_top():
    kind, text = ev.verdict_only_a(_twin_rows())
    assert kind == "warning" and "bis ρ = 0,9 bleibt B bei höchstens 0,002" in text and "**0,50**" in text and "A fällt von 1,87 auf 0,99" in text


def test_verdict_outside_grid_and_remedies():
    rows = [{"rho": 0.0, "measured": 0.0106, "exact": 0.0111}, {"rho": 0.99, "measured": 0.768, "exact": 0.7709}]
    kind, text = ev.verdict_outside_grid(rows, 0.99)
    assert kind == "warning" and "**76,8 %**" in text and "exakt 77,1 %" in text
    assert ev.verdict_outside_grid(rows, 0.0)[0] == "info"
    kind, text = ev.verdict_remedies(_twin_rows(), 0.99)
    assert kind == "info" and "A **0,70** und B **0,72**" in text and "als Gruppe **1,87**" in text and "A **0,04** und B **0,04**" in text and "100,0 %" in text and "0,90" in text


def test_helper_formats():
    assert ev._pct(0.1234) == "12,3 %" and ev._num(0.5, 2) == "0,50" and ev._rho(0.99) == "0,99" and ev._rho(0.0) == "0"
    assert all(len(v) > 40 for v in C.PRESET_HELP.values())
