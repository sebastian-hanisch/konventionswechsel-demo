"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, gerechnete Aussagen der Hilfetexte."""

import math

import pytest

import kvw_constants as C
import kvw_evaluation as V
import kvw_exact as E
import kvw_presets as P
import kvw_sim as SI


def _settings(name):
    return V.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert V.Settings(**C.PRESETS[first]) == V.Settings()
    assert first.startswith("Offene Börse")


def test_setting_specs_cover_every_preset_key_and_permalink_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_open_exchange_preset_is_far_above_the_simulation_budget():
    """„liegt weit über dem Budget der Simulation“: n = 20, q = 0,4, ε = 0,1 → 10^4,95 Aktualisierungen je Betrieb gegen 5 000 (100 000 Schritte / 20)."""
    view = V.analyse_main(_settings("Offene Börse (Vollgraph, n = 20)"))
    assert view.exceeds_budget and view.log10_time == pytest.approx(4.95, abs=0.01) and view.budget_time == pytest.approx(5000)
    assert view.log10_time - math.log10(view.budget_time) > 1


def test_corridor_preset_reaches_the_goal_in_the_simulation():
    """„Die Simulation erreicht das Ziel“: Ring n = 80, k = 2, q = 0,4, ε = 0,1: alle 60 Läufe am Ziel innerhalb der Schrittgrenze, Mittel 8 bis 20."""
    s = _settings("Korridor (Ring, k = 2, n = 80)")
    view = V.analyse_main(s)
    assert (view.spec.d, view.spec.thr, view.spec.edge, view.spec.spread) == (4, 2, 2, 0)
    sm = V.run_main_simulation(view)
    assert sm.capped == 0 and 8 < sm.mean < 20


def test_regions_preset_reaches_the_goal_and_the_edge_advances():
    s = _settings("Regionen (Gitter 12 × 12)")
    view = V.analyse_main(s)
    assert (view.spec.n, view.spec.d, view.spec.thr, view.spec.edge, view.spec.spread) == (144, 4, 1, 1, 0)
    sm = V.run_main_simulation(view)
    assert sm.capped == 0 and sm.mean < 10


def test_near_complete_ring_preset_sees_28_of_29():
    s = _settings("Fast jeder sieht jeden (Ring, k = 14, n = 30)")
    view = V.analyse_main(s)
    assert (view.spec.d, view.spec.n, s.eps) == (28, 30, 0.2) and view.spec.d / (view.spec.n - 1) == pytest.approx(28 / 29)


def test_back_preset_has_a_higher_threshold_and_a_larger_exponent():
    s = _settings("Rückwechsel (Vollgraph, n = 40)")
    back = V.analyse_main(s)
    forth = V.analyse_main(V.Settings(**{**C.PRESETS["Rückwechsel (Vollgraph, n = 40)"], "direction": C.DIRECTIONS[0]}))
    assert back.back and not forth.back and back.thr > forth.thr
    assert back.log10_time / forth.log10_time == pytest.approx(2.06, abs=0.05)
    assert back.log10_time == pytest.approx(E.mirror_log_time(40, forth.thr, s.eps) / E.LN10)


def test_exact_time_and_simulation_of_the_preset_with_a_finite_exact_value_agree():
    """Ring n = 80 hat keine exakte Zeit (2^80); die Prüfung gegen exakt läuft in den Orakel-Tests. Hier nur: die Simulation der Presets ist bei gleichem Seed gleich."""
    view = V.analyse_main(_settings("Korridor (Ring, k = 2, n = 80)"))
    assert V.run_main_simulation(view).mean == V.run_main_simulation(view).mean


def test_simulation_streams_of_presets_are_deterministic():
    s = _settings("Regionen (Gitter 12 × 12)")
    a = SI.stream_seed(s.seed, 0)
    assert a == SI.stream_seed(C.DEFAULT_SEED, 0)
