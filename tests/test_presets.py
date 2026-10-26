"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Hilfetexte, Adresszeile."""

import pytest

import kmk_constants as C
import kmk_evaluation as ev
import kmk_presets as P


def test_presets_are_complete():
    assert "Standardfall (Voreinstellung)" in C.PRESETS
    assert set(C.PRESET_HELP) == set(C.PRESETS)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name] and C.PRESET_HELP[name] != name         # ein echter Text, nicht der Name
        assert set(preset) == set(P.PRESET_KEYS), name
        assert (preset["world"] in C.WORLDS and preset["rho"] in C.RHOS and C.N_MIN <= preset["n"] <= C.N_MAX and C.TREES_MIN <= preset["trees"] <= C.TREES_MAX
                and C.NOISE_MIN <= preset["noise"] <= C.NOISE_MAX and C.THRESHOLD_MIN <= preset["threshold"] <= C.THRESHOLD_MAX and 0 <= preset["seed"] <= C.SEED_MAX), name


def test_the_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    keys = ("world_select", "rho_select", "n_slider", "trees_slider", "noise_slider", "threshold_slider", "seed_input")
    assert [P.SETTING_SPECS[k].default for k in keys] == [s.world, s.rho, s.n, s.trees, s.noise, s.threshold, s.seed]


def test_every_preset_differs_from_every_other():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_the_presets_that_name_a_world_use_one_of_the_grid_values():
    for name in ("Dieselbe Information, springende Aufteilung", "Nur A trägt bei", "Ohne Zwilling", "Mischen erzeugt Unmögliches"):
        assert C.PRESETS[name]["rho"] in C.GRID_RHOS, name
    assert C.PRESETS["Nur A trägt bei"]["world"] == "nur_a"


@pytest.mark.parametrize("raw,expected", [("99999", C.N_MAX), ("1", C.N_MIN), ("700", 700), ("abc", None), ("", None)])
def test_clean_value_clamps_integers(raw, expected):
    assert P.clean_value(P.SETTING_SPECS["n_slider"], raw) == expected


def test_clean_value_accepts_only_known_choices_and_clamps_the_rest():
    assert P.clean_value(P.SETTING_SPECS["world_select"], "nur_a") == "nur_a" and P.clean_value(P.SETTING_SPECS["world_select"], "xyz") is None
    assert P.clean_value(P.SETTING_SPECS["rho_select"], "0.99") == 0.99 and P.clean_value(P.SETTING_SPECS["rho_select"], "0.98") is None
    thr = P.SETTING_SPECS["threshold_slider"]
    assert P.clean_value(thr, "0.5") == C.THRESHOLD_MIN and P.clean_value(thr, "2") == C.THRESHOLD_MAX and P.clean_value(thr, "0.8765") == 0.88
    assert P.clean_value(P.SETTING_SPECS["seed_input"], "-5") == 0 and P.clean_value(P.SETTING_SPECS["seed_input"], str(C.SEED_MAX + 1)) == C.SEED_MAX
