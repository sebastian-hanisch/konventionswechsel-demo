"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12, Punkt 8): Schwellen und jede Zahl im Text an handgesetzten Werten; Formatierung."""

import math

import numpy as np
import pytest

import kvw_constants as C
import kvw_evaluation as V
import kvw_scenario as S
from kvw_evaluation import SimSummary


def summary(**kw):
    base = dict(runs=60, finished=60, capped=0, mean=12.0, se=0.5, median=10.0, p10=3.0, p90=25.0, cap_time=5000.0, exact=None, z=None)
    base.update(kw)
    return SimSummary(**base)


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def test_de_and_thousands():
    assert V.de(1.5, 1) == "1,5" and V.de(3.14159, 3) == "3,142" and V.de(7, 0) == "7"
    assert V.thousands(1234567) == "1.234.567" and V.thousands(999.6) == "1.000" and V.thousands(0) == "0"


@pytest.mark.parametrize("l10,text", [(-1.4, "0,04"), (-0.3, "0,50"), (0.0, "1,0"), (1.0, "10,0"), (1.5, "31,6"), (2.0, "100"), (3.0, "1.000"), (4.0, "10.000"), (5.0, "1,0·10^5"), (9.62, "4,2·10^9"), (99.99, "9,8·10^99"),
                                      (99.999, "1,0·10^100"), (100.0, "10^100,0"), (385.55, "10^385,6")])
def test_fmt_log10(l10, text):
    assert V.fmt_log10(l10) == text


def test_fmt_time_handles_nan_and_zero():
    assert V.fmt_time(float("nan")) == "–" and V.fmt_time(0.0) == "–" and V.fmt_time(12.0) == "12,0" and V.fmt_time(0.04) == "0,04" and V.fmt_time(30000.0) == "30.000"


def test_fmt_years_uses_weeks_and_switches_to_months():
    assert V.fmt_years(math.log10(52 * 10)) == "10,0 Jahre"
    assert V.fmt_years(math.log10(26)) == "6,0 Monate"                  # 26 Wochen = 0,5 Jahre = 6 Monate
    assert V.fmt_years(math.log10(496)) == "9,5 Jahre"                              # 496 Aktualisierungen / 52 = 9,54 Jahre
    assert V.fmt_years(77.0).endswith("Jahre") and "10^" in V.fmt_years(110.0)


# --- Hauptansicht ----------------------------------------------------------------------------------------------------------------------------
def test_verdict_main_complete_graph_inside_the_budget():
    view = V.analyse_main(V.Settings(n=10, eps=0.3, budget=300_000))
    kind, text = V.verdict_main(view)
    assert kind == "success" and not view.exceeds_budget
    assert text.startswith("**Hinwechsel (alle B → A) im Vollgraph (n = 10, jeder trifft jeden):** erwartet ")
    assert f"erwartet {V.fmt_log10(view.log10_time)} Aktualisierungen" in text and "Dazu müssen 4 von 9 Partnern schon ihr Ziel führen" in text
    assert "(300.000 Schritte = 30.000 Aktualisierungen je Betrieb)" in text and "innerhalb der Schrittgrenze" in text and "Prüfung je Betrieb und Woche" in text


def test_verdict_main_complete_graph_above_the_budget():
    view = V.analyse_main(V.Settings())                 # n = 20, ε = 0,1: 10^4,95 gegen 5 000
    kind, text = V.verdict_main(view)
    assert kind == "warning" and view.exceeds_budget and "erwartet 89.448 Aktualisierungen" in text
    assert "(100.000 Schritte = 5.000 Aktualisierungen je Betrieb): Sie erreicht das Ziel nicht." in text and "Dazu müssen 8 von 19 Partnern" in text


def test_verdict_main_back_direction_names_the_mirrored_threshold():
    view = V.analyse_main(V.Settings(n=40, direction=C.DIRECTIONS[1]))
    kind, text = V.verdict_main(view)
    assert view.back and view.thr == 24 and view.target == 26                 # n = 40: thr = 16, thr_B = 40 − 16 = 24, Ziel 26
    assert text.startswith("**Rückwechsel (alle A → B) im Vollgraph (n = 40") and "Dazu müssen 24 von 39 Partnern" in text and kind == "warning"


def test_verdict_main_other_networks_point_to_the_simulation():
    view = V.analyse_main(V.Settings(network=C.NETWORKS[1], n=80, k=2))
    kind, text = V.verdict_main(view)
    assert kind == "info" and view.log10_time is None and view.exceeds_budget is None
    assert "(n = 80, 4 Partner, nötig sind 2)" in text and "über alle 2^n Belegungen" in text and "Rückwechsel" not in text


def test_verdict_spread_complete_graph():
    kind, text = V.verdict_spread(S.resolve_network(C.NETWORKS[0], 20, 2, 0.4))
    assert kind == "info" and "braucht 8 von 19 Partnern auf A (42 %)" in text and "keinen Rand" in text


def test_verdict_spread_advancing_edge():
    kind, text = V.verdict_spread(S.resolve_network(C.NETWORKS[1], 80, 2, 0.4))
    assert kind == "success" and "sieht 2 A-Partner, nötig sind 2 (Spielraum s = 0)" in text
    kind, text = V.verdict_spread(S.resolve_network(C.NETWORKS[1], 80, 5, 0.4))
    assert kind == "success" and "sieht 5 A-Partner, nötig sind 4 (Spielraum s = 1)" in text


def test_verdict_spread_edge_that_does_not_advance():
    kind, text = V.verdict_spread(S.resolve_network(C.NETWORKS[3], 144, 2, 0.4))
    assert kind == "warning" and "sieht nur 3 A-Partner, nötig sind 4 (Spielraum s = -1)" in text and "nicht vor" in text


# --- Simulation ------------------------------------------------------------------------------------------------------------------------------
def test_summarise_ignores_capped_runs_and_computes_z():
    times = np.array([2.0, 4.0, np.nan, 6.0])
    sm = V.summarise(times, 400, 20, exact=3.0)
    assert (sm.runs, sm.finished, sm.capped) == (4, 3, 1) and sm.mean == pytest.approx(4.0) and sm.cap_time == pytest.approx(20.0)
    assert sm.se == pytest.approx(2.0 / math.sqrt(3)) and sm.z == pytest.approx((4.0 - 3.0) / sm.se) and sm.median == 4.0
    none = V.summarise(np.array([np.nan, np.nan]), 10, 5)
    assert none.finished == 0 and math.isnan(none.mean) and math.isnan(none.median) and none.z is None


def test_verdict_simulation_all_finished_without_exact():
    kind, text = V.verdict_simulation(summary())
    assert kind == "info" and "Simulation, 60 Läufe, alle am Ziel" in text and "Mittel 12,0 ± 0,50" in text and "Median 10,0, 10 % der Läufe unter 3,0, 10 % über 25,0" in text and not "streut stark" in text


def test_verdict_simulation_against_exact_inside_and_outside_the_margin():
    kind, text = V.verdict_simulation(summary(exact=11.0, z=2.0))
    assert kind == "success" and "Exakt erwartet: 11,0; Abweichung z = 2,0 Standardfehler (Marge 4)" in text
    kind, text = V.verdict_simulation(summary(exact=11.0, z=-4.0))                  # Grenze inklusive
    assert kind == "success" and "z = -4,0" in text
    kind, text = V.verdict_simulation(summary(exact=11.0, z=4.1))
    assert kind == "warning" and "außerhalb der Marge von 4" in text


def test_verdict_simulation_partial_and_empty():
    kind, text = V.verdict_simulation(summary(finished=57, capped=3, mean=900.0))
    assert kind == "warning" and "3 von 60 Läufen erreichten das Ziel nicht innerhalb der Schrittgrenze (5.000 Aktualisierungen je Betrieb)" in text and "nach unten verzerrt" in text and "(900" in text
    kind, text = V.verdict_simulation(summary(finished=0, capped=60, mean=float("nan"), exact=1e6))
    assert kind == "warning" and text.startswith("Kein Lauf von 60 erreichte das Ziel") and "Exakt erwartet: 1,0·10^6." in text


def test_verdict_check_counts_cells_inside_the_margin():
    rows = [{"z": 1.0, "capped": 0, "runs": 120}, {"z": -3.9, "capped": 0, "runs": 120}, {"z": 4.0, "capped": 0, "runs": 120}]
    kind, text = V.verdict_check(rows)
    assert kind == "success" and text.startswith("3 von 3 Zellen liegen innerhalb von 4 Standardfehlern") and "größte Abweichung |z| = 4,0; 360 Läufe" in text
    kind, text = V.verdict_check(rows[:2] + [{"z": 4.5, "capped": 0, "runs": 120}])
    assert kind == "warning" and text.startswith("2 von 3")
    kind, text = V.verdict_check([{"z": 0.5, "capped": 2, "runs": 120}])
    assert kind == "warning" and "2 Läufe erreichten die Schrittgrenze nicht" in text


# --- Experimente -----------------------------------------------------------------------------------------------------------------------------
def cell(mean, capped=0):
    return {"mean": mean, "se": 0.1, "capped": capped, "total": 240}


def e1_rows(k1, k2, k3, k4):
    ns = [10, 20, 40, 80, 160, 320]
    return [{"n": n, "complete_log10": 2.7 + 0.24 * (n - 10), "rings": {1: cell(k1[i]), 2: cell(k2[i]), 3: cell(k3[i]), 4: cell(k4[i])}} for i, n in enumerate(ns)]


def test_spread_percent_and_ring_flatness():
    assert V.spread_percent([2.0, 2.2, 2.1]) == pytest.approx(10.0)
    rows = e1_rows([5, 4.0, 4.1, 4.2, 4.1, 4.0], [20, 12, 10.0, 10.5, 10.4, 10.2], [90, 70, 50, 40, 35, 30], [900, 700, 500, 400, 300, 200])
    assert V.ring_flatness(rows, 1, 20) == pytest.approx(5.0) and V.ring_flatness(rows, 2, 40) == pytest.approx(5.0)
    assert V.ring_flatness(rows[:3], 2, 40) is None


def test_verdict_e1_text():
    rows = e1_rows([5, 4.0, 4.1, 4.2, 4.1, 4.0], [20, 12, 10.0, 10.5, 10.4, 10.2], [90, 70, 50, 40, 35, 30], [900, 700, 500, 400, 300, 200])
    kind, text = V.verdict_e1(rows, 0.1)
    assert kind == "info" and text.startswith("**ε = 0,10:** im Vollgraph wächst die Zeit von 501 (n = 10) auf")
    assert "k = 1 ändert sich die Zeit ab n = 20 um höchstens 5,0 %, mit k = 2 ab n = 40 um höchstens 5,0 %" in text
    assert "Bei k = 3 fällt die Zeit von n = 40 auf n = 320 noch auf das 0,60-fache" in text and "Bei k = 4 fällt die Zeit von n = 40 auf n = 320 noch auf das 0,40-fache" in text
    empty = [{"n": n, "complete_log10": 2.0, "rings": {k: None for k in (1, 2, 3, 4)}} for n in (10, 20)]
    assert "fehlen" in V.verdict_e1(empty, 0.1)[1]


def test_verdict_e2_text():
    rate_rows = [{"eps": 0.2, "measured": 0.1350, "chernoff": 0.1352, "rough": 0.28, "rough_over_measured": 2.07, "diff": -0.0002},
                 {"eps": 0.1, "measured": 0.2413, "chernoff": 0.2415, "rough": 0.4, "rough_over_measured": 1.66, "diff": 0.0001}]
    exp_rows = [{"n": 10, "thr": 4, "a": 3.67, "a_over_thr": 0.92}, {"n": 80, "thr": 32, "a": 30.11, "a_over_thr": 0.94}]
    kind, text = V.verdict_e2(rate_rows, exp_rows, 0.4)
    assert kind == "success" and "höchstens um 0,0002 von KL" in text and "das 1,66- bis 2,07-fache" in text and "a liegt bei 0,92 bis 0,94 mal thr (n = 10 bis 80)" in text and text.startswith("**q = 0,40:**")


def e3_row(k, mean, ratio, share, capped=0):
    return {"k": k, "d": 2 * k, "share": share, "thr": 2, "mean": mean, "se": 0.1, "capped": capped, "total": 160, "ratio": ratio, "complete": 1000.0}


def test_verdict_e3_text_end_sawtooth_and_slower_ring():
    rows = [e3_row(1, 3.0, 300.0, 0.1), e3_row(2, 8.0, 100.0, 0.3), e3_row(3, 4.0, 200.0, 0.5), e3_row(4, 600.0, 1.5, 0.9), e3_row(5, 1500.0, 0.66, 0.97)]
    assert V.first_advantage_end(rows)["k"] == 4 and V.sawtooth_steps(rows) == [(2, 3)]
    kind, text = V.verdict_e3(rows, 30, 0.2)
    assert kind == "info" and text.startswith("**n = 30, ε = 0,20:** Bei k = 1 (sieht 10 % der anderen) ist der Ring um den Faktor 300 schneller")
    assert "bei etwa halber Sicht (k = 3, 50 %) noch um den Faktor 200" in text and "erst bei k = 4 (sieht 90 %) auf höchstens das Doppelte (1,50)" in text
    assert "Bei k = 5 (sieht 97 %) ist der Ring langsamer als der Vollgraph (Faktor 0,66)" in text and "nicht monoton in k: bei k = 2 → 3 fällt sie" in text
    assert V.first_advantage_end(rows[:2]) is None
    kind, text = V.verdict_e3(rows[:2] + [e3_row(9, 90.0, 0.98, 0.95, capped=19)], 20, 0.1)
    assert kind == "warning" and "19 Läufe erreichten die Schrittgrenze nicht" in text
    assert V.verdict_e3([], 20, 0.1)[0] == "info"


def e4_row(k, q, thr, mean):
    return {"k": k, "q": q, "thr": thr, "spread": k - thr, "mean": mean, "se": 0.1, "capped": 0, "total": 240}


def test_verdict_e4_text_factors_and_spread():
    rows = [e4_row(3, 0.3, 2, 4.0), e4_row(3, 0.4, 3, 14.0), e4_row(4, 0.4, 4, 36.0), e4_row(4, 0.3, 3, 7.0), e4_row(5, 0.35, 4, 16.0), e4_row(5, 0.45, 5, 117.0), e4_row(6, 0.3, 4, 9.0), e4_row(3, 0.45, 3, 14.0)]
    assert V.spread_zero_factors(rows) == [(3, 4, pytest.approx(36 / 14)), (4, 5, pytest.approx(117 / 36))]
    kind, text = V.verdict_e4(rows)
    assert kind == "info" and "Bei thr = 4 fällt sie mit dem Spielraum s = k − thr: s = 0: 36,0, s = 1: 16,0, s = 2: 9,0." in text
    assert "um den Faktor 2,6, 3,2." in text and "zahlengleich" in text
    assert V.verdict_e4([e4_row(3, 0.3, 2, None)])[1] == "Die vorgerechneten Zellen fehlen."


def e5_row(kind, q, thr, eps, side, mean, capped=0):
    return {"kind": kind, "q": q, "thr": thr, "edge": 1 if kind == "vn" else 3, "eps": eps, "side": side, "mean": mean, "se": 0.1, "capped": capped, "total": 240}


def test_grid_flatness_and_verdict_e5():
    rows = [e5_row("vn", 0.25, 1, 0.2, 12, 1.8), e5_row("vn", 0.25, 1, 0.2, 24, 1.9), e5_row("moore", 0.3, 3, 0.1, 12, 10.0), e5_row("moore", 0.3, 3, 0.1, 24, 9.0),
            e5_row("moore", 0.4, 4, 0.1, 12, 5000.0, 1), e5_row("moore", 0.4, 4, 0.1, 24, 1200.0, 3)]
    flat = V.grid_flatness(rows)
    assert [(f["kind"], f["thr"], round(f["change"], 1)) for f in flat] == [("moore", 3, -10.0), ("moore", 4, -76.0), ("vn", 1, 5.6)]
    kind, text = V.verdict_e5(rows)
    assert kind == "warning" and "In 2 Zellen (Schwelle höchstens 3) ändert sich die Zeit von L = 12 auf L = 24 um höchstens 10,0 %" in text and "4 Läufe in den beiden größten Gittern" in text and "um -76 % und wird nicht gedeutet" in text
    kind, text = V.verdict_e5(rows[:4])
    assert kind == "info" and "In 2 Zellen" in text and "nicht gesättigt" not in text
    assert V.verdict_e5([])[0] == "info"


def test_verdict_e6_text():
    rows = [{"eps": 0.2, "n": 640, "ratio": 2.42}, {"eps": 0.1, "n": 640, "ratio": 2.06}, {"eps": 0.05, "n": 640, "ratio": 1.90}, {"eps": 0.1, "n": 20, "ratio": 2.0}]
    pair = {"slope_hin": 0.2410, "slope_back": 0.4968, "rate_hin": 0.2415, "rate_back": 0.4972}
    kind, text = V.verdict_e6(rows, pair, 0.4)
    assert kind == "info" and "Steigung hin 0,2410 (KL(q ‖ ε/2)/ln 10 = 0,2415), zurück 0,4968 (KL(1 − q ‖ ε/2)/ln 10 = 0,4972)" in text
    assert "bei n = 640: ε = 0,20: 2,42, ε = 0,10: 2,06, ε = 0,05: 1,90." in text


def test_verdict_e7_text_and_kinds():
    rows = [{"n": 10, "deviation": 1e-14, "negative": False}, {"n": 30, "deviation": 3e-9, "negative": False}, {"n": 60, "deviation": 0.019, "negative": False}, {"n": 70, "deviation": 1.1, "negative": True}]
    kind, text = V.verdict_e7(rows)
    assert kind == "warning" and "Bis n = 30 stimmt das lineare System mit dem Logarithmus auf relativ 3,0e-09 überein" in text
    assert "Ab n = 60 weicht es um mehr als 1 % ab, bei n = 70 liefert es eine negative Zeit." in text and "Plattform" in text
    kind, text = V.verdict_e7(rows[:2])
    assert kind == "info" and "Ab n" not in text
    assert "weicht schon bei kleinem n ab" in V.verdict_e7([{"n": 10, "deviation": 0.5, "negative": False}])[1]
