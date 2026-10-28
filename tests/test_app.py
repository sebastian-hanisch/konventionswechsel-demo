"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine
Grenzen und prüft, dass kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=)
und `StreamlitAPIException` bei Reglern mit berechneten Grenzen. Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import kvw_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
TAB_LABELS = ["1 Zeit über n", "2 Welche Rate?", "3 Wie groß darf die Nachbarschaft sein?", "4 Die Treppe der Schwelle", "5 Gitter", "6 Rückwechsel", "7 Warum im Logarithmus"]


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info])


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue                       # der Regler k ist im Vollgraph ausgegraut (eigener Test unten)
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("network", C.NETWORKS)
def test_all_sliders_at_their_minima_and_all_at_their_maxima_do_not_raise(network):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): alle Regler zugleich am Minimum, dann alle am Maximum (n = 1 000, k = 14, q = 0,1 …), je Netz; im Vollgraph auch beide Richtungen.
    Ausgegraute Regler (k außerhalb des Rings, Richtung außerhalb des Vollgraphs) lassen sich nicht setzen und bleiben auf ihrem Wert."""
    for edge in ("min", "max"):
        for direction in (C.DIRECTIONS if network == C.NETWORKS[0] else (C.DIRECTIONS[0],)):
            at = _fresh_app()
            at.selectbox(key="network_select").set_value(network).run()
            for slider in at.sidebar.slider:
                if slider.disabled:
                    continue
                value = slider.min if edge == "min" else slider.max
                slider.set_value(int(value) if isinstance(slider.value, int) else value)
            if network == C.NETWORKS[0]:
                at.selectbox(key="direction_select").set_value(direction)
            at.select_slider(key="budget_select").set_value(C.SIM_BUDGETS[0] if edge == "min" else C.SIM_BUDGETS[-1])
            at.run()
            assert not at.exception, (edge, network, direction, [str(e) for e in at.exception])


def test_the_two_play_buttons_work_individually():
    at = _fresh_app()
    assert not at.session_state["sim_signature"] if "sim_signature" in at.session_state else True
    at.button(key="sim_button").click().run()
    assert not at.exception and at.session_state["sim_signature"] is not None
    assert "erreichten das Ziel nicht innerhalb der Schrittgrenze" in _texts(at) or "Kein Lauf von" in _texts(at)           # Vollgraph n = 20 liegt über der Schrittgrenze

    at = _fresh_app()
    assert not any("Simulation (Mittel ± Standardfehler)" in df.value.columns.tolist() for df in at.dataframe)
    at.button(key="check_button").click().run()
    assert not at.exception and at.session_state["check_done"] is True
    assert any("Simulation (Mittel ± Standardfehler)" in df.value.columns.tolist() for df in at.dataframe) and "Zellen liegen innerhalb" in _texts(at)


def test_simulation_is_hidden_when_the_settings_change():
    at = _fresh_app()
    at.button(key="sim_button").click().run()
    assert "Schrittgrenze (" in _texts(at)
    at.slider(key="n_slider").set_value(30).run()
    assert not at.exception and "Schrittgrenze (" not in _texts(at)


def test_corridor_preset_simulation_reaches_the_goal():
    at = _fresh_app()
    at.button(key="preset_Korridor (Ring, k = 2, n = 80)").click().run()
    at.button(key="sim_button").click().run()
    assert not at.exception and "alle am Ziel" in _texts(at)


def test_back_preset_shows_the_mirrored_verdict():
    at = _fresh_app()
    at.button(key="preset_Rückwechsel (Vollgraph, n = 40)").click().run()
    assert not at.exception and "Rückwechsel (alle A → B) im Vollgraph" in _texts(at)


def test_k_slider_is_disabled_outside_the_ring_and_direction_outside_the_complete_graph():
    at = _fresh_app()
    assert at.sidebar.slider(key="k_slider").disabled is True and at.sidebar.selectbox(key="direction_select").disabled is False
    at.selectbox(key="network_select").set_value(C.NETWORKS[1]).run()
    assert at.sidebar.slider(key="k_slider").disabled is False and at.sidebar.selectbox(key="direction_select").disabled is True


def test_grid_rounding_and_ring_clamp_are_announced():
    at = _fresh_app()
    at.selectbox(key="network_select").set_value(C.NETWORKS[2]).run()
    at.slider(key="n_slider").set_value(150).run()
    assert not at.exception and any("auf 144 (L = 12) gerundet" in c.value for c in at.caption)
    at = _fresh_app()
    at.selectbox(key="network_select").set_value(C.NETWORKS[1]).run()
    at.slider(key="n_slider").set_value(6).run()
    at.slider(key="k_slider").set_value(14).run()
    assert not at.exception and any("k wurde auf 2 begrenzt" in c.value for c in at.caption)


def test_default_state_shows_the_seven_tabs_and_the_main_verdict():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == TAB_LABELS
    assert "Hinwechsel (alle B → A) im Vollgraph (n = 20" in _texts(at)


def test_permalink_sets_the_network_and_is_limited_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["net"] = C.NETWORKS[1]
    at.query_params["n"] = "0"
    at.query_params["k"] = "99"
    at.query_params["q"] = "0.9"
    at.query_params["budget"] = "300000"
    at.run()
    assert not at.exception
    assert at.selectbox(key="network_select").value == C.NETWORKS[1] and at.slider(key="n_slider").value == C.N_MIN and at.slider(key="k_slider").value == C.K_MAX
    assert at.slider(key="q_slider").value == C.Q_MAX and at.select_slider(key="budget_select").value == 300000


def test_permalink_ignores_unknown_values_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["net"] = "Stern"
    at.query_params["n"] = "viel"
    at.query_params["budget"] = "12345"
    at.query_params["q"] = "nan"
    at.run()
    assert not at.exception
    assert at.selectbox(key="network_select").value == C.NETWORKS[0] and at.slider(key="n_slider").value == C.DEFAULT_N and at.select_slider(key="budget_select").value == C.DEFAULT_BUDGET
    assert at.slider(key="q_slider").value == C.DEFAULT_Q


def test_pdf_button_creates_a_pdf_and_offers_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_every_preset_shows_its_main_verdict_of_the_expected_kind():
    expected = {"Offene Börse (Vollgraph, n = 20)": ("warning", "Hinwechsel"), "Korridor (Ring, k = 2, n = 80)": ("success", "gerade Kante rückt ohne Fehler vor"),
                "Regionen (Gitter 12 × 12)": ("success", "gerade Kante rückt ohne Fehler vor"), "Fast jeder sieht jeden (Ring, k = 14, n = 30)": ("success", "gerade Kante rückt ohne Fehler vor"),
                "Rückwechsel (Vollgraph, n = 40)": ("warning", "Rückwechsel")}
    for name, (kind, word) in expected.items():
        at = _fresh_app()
        at.button(key=f"preset_{name}").click().run()
        assert not at.exception, name
        texts = {"warning": [w.value for w in at.warning], "success": [s.value for s in at.success], "info": [i.value for i in at.info]}[kind]
        assert any(word in t for t in texts), (name, kind)
