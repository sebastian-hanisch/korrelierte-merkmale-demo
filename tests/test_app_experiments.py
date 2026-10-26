"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kmk_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["world_select"] == preset["world"] and at.session_state["rho_select"] == preset["rho"] and at.session_state["n_slider"] == preset["n"]
    assert at.session_state["threshold_slider"] == preset["threshold"] and at.session_state["noise_slider"] == preset["noise"]
    assert len(at.get("plotly_chart")) == 9                   # Hauptansicht 3 (echt, gemischt, Balken), E1 bis E5: 1 + 2 + 1 + 1 + 1 = 6


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["world"] = "nur_a"
    at.query_params["rho"] = "0.99"
    at.query_params["n"] = "99999"
    at.query_params["trees"] = "1"
    at.query_params["noise"] = "abc"                # ungültig: wird ignoriert
    at.query_params["thr"] = "0.123"
    at.run()
    assert not at.exception
    assert at.session_state["world_select"] == "nur_a" and at.session_state["rho_select"] == 0.99 and at.session_state["n_slider"] == C.N_MAX
    assert at.session_state["trees_slider"] == C.TREES_MIN and at.session_state["noise_slider"] == C.DEFAULT_NOISE and at.session_state["threshold_slider"] == C.THRESHOLD_MIN


def test_permalink_ignores_an_unknown_world_and_rho():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["world"] = "xyz"
    at.query_params["rho"] = "0.5000001"
    at.run()
    assert not at.exception and at.session_state["world_select"] == "beide" and at.session_state["rho_select"] == C.DEFAULT_RHO


def test_six_metrics_show_the_forest_numbers():
    at = _app()
    assert [m.label for m in at.metric] == ["A einzeln gemischt", "B einzeln gemischt", "Anteil von A an A + B", "A + B als Gruppe", "A bedingt auf B", "Außerhalb nach dem Mischen"]
    assert all(m.value for m in at.metric) and at.metric[2].value.endswith("%")


def test_the_headline_verdict_depends_on_the_world():
    at = _app()
    assert "Aufteilung A : B" in _texts(at)
    at.sidebar.selectbox(key="world_select").set_value("nur_a").run()
    assert not at.exception and "Nur A trägt bei" in _texts(at) or "B trägt nichts bei" in _texts(at)


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e2_rho", 0.0), ("e2_rho", 0.999), ("e4_rho", 0.5), ("e4_rho", 0.999), ("e5_rho", 0.0), ("e5_rho", 0.9)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Beide tragen bei: die Summe der Einzelwichtigkeiten ist", "der Anteil von A schwankt von Datensatz zu Datensatz", "mehr Bäume beruhigen sie nicht", "Die Verwässerung gilt nur, wenn beide Zwillinge Information tragen",
                   "nach dem Mischen von A liegen gezählt", "einzeln gemischt A"):
        assert needle in text, needle


def test_every_slider_corner_runs():
    for edge in ("min", "max"):
        at = _app()
        for key in ("n_slider", "trees_slider", "noise_slider", "threshold_slider"):
            w = at.sidebar.slider(key=key)
            value = w.min if edge == "min" else w.max
            w.set_value(int(value) if isinstance(w.value, int) else value)
        at.sidebar.select_slider(key="rho_select").set_value(C.RHOS[0] if edge == "min" else C.RHOS[-1])
        at.run()
        assert not at.exception, (edge, [str(e) for e in at.exception])
        assert all(m.value for m in at.metric)


def test_a_missing_grid_shows_a_warning_instead_of_failing(monkeypatch, tmp_path):
    import streamlit as st

    import kmk_evaluation as ev

    st.cache_data.clear()                                  # zwischengespeicherte Raster aus anderen Tests verwerfen
    monkeypatch.setattr(ev, "GRID_PATH", tmp_path / "gibt-es-nicht.json")
    monkeypatch.setattr(ev, "load_grid", lambda path=tmp_path / "gibt-es-nicht.json": ev.json.loads("null"))
    try:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        assert not at.exception, [str(e) for e in at.exception]
        assert any("Das vorgerechnete Raster" in w.value for w in at.warning)
    finally:
        st.cache_data.clear()
