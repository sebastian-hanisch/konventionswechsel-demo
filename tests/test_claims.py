"""Jede Zahl aus der README, nachgerechnet über dieselben Funktionen wie in der App (DEMO-PLAYBOOK §4). Simulationen: die vorgerechneten Zellen (vier feste Zufalls-Ströme); Rundungen wie in der README."""

import math
from pathlib import Path

import pytest

import kvw_constants as C
import kvw_evaluation as V
import kvw_exact as E
import kvw_experiments as X
import kvw_grid as GRID
import kvw_model as M

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cells():
    return X.load_measurements()["cells"]


def ring(cells, n, k, eps, q=0.4):
    return X.find_cell(cells, "ring", n, k, M.threshold(q, 2 * k), eps)


def r1(x):
    return round(x, 1)


# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------
def test_context_weeks_and_years(cells):
    """„89.448 Aktualisierungen, rund 1.720 Jahre“ (n = 20, q = 0,4, ε = 0,1; 52 Wochen je Jahr); Ring k = 2: 12,0 bis 12,7 Aktualisierungen (n = 40 bis 320)."""
    t = 10 ** X.complete_log10(20, 0.4, 0.1)
    assert round(t) == 89448 and round(t / C.WEEKS_PER_YEAR, -1) == 1720
    means = [ring(cells, n, 2, 0.1)["mean"] for n in (40, 80, 160, 320)]
    assert r1(min(means)) == 12.0 and r1(max(means)) == 12.7


# --- H1 --------------------------------------------------------------------------------------------------------------------------------------
def test_h1_exponential_growth_in_the_complete_graph():
    values = [X.complete_log10(n, 0.4, 0.1) for n in (10, 20, 40, 80, 160, 320)]
    assert [V.fmt_log10(v) for v in values] == ["496", "89.448", "4,2·10^9", "1,3·10^19", "2,0·10^38", "6,1·10^76"]
    assert X.complete_log10(1600, 0.4, 0.1) == pytest.approx(385.5, abs=0.05)
    assert all(b > a for a, b in zip(values, values[1:]))


def test_h1_hand_values_of_the_mini_instances(mini3, ring4):
    assert mini3["time"][0.2] == pytest.approx(5.5556, abs=5e-5) and mini3["time"][0.1] == pytest.approx(10.5263, abs=5e-5) and ring4["time"] == pytest.approx(4.2105, abs=5e-5)


# --- H2, H2b ---------------------------------------------------------------------------------------------------------------------------------
def test_h2_ring_k1_and_k2_are_nearly_flat_in_n(cells):
    spans1, spans2 = [], []
    for eps in (0.2, 0.1, 0.05):
        rows = X.e1_rows(cells, eps)
        spans1.append(V.ring_flatness(rows, 1, 20))
        spans2.append(V.ring_flatness(rows, 2, 40))
    assert [r1(s) for s in spans1] == [4.4, 2.6, 1.7] and [r1(s) for s in spans2] == [1.6, 6.2, 12.6]
    assert all(c["capped"] == 0 for c in cells if c["kind"] == "ring" and c["k"] in (1, 2) and "e1" in c["experiments"])
    assert (r1(ring(cells, 40, 2, 0.05)["mean"]), r1(ring(cells, 320, 2, 0.05)["mean"])) == (25.5, 22.6)


def test_h2b_ring_k3_and_k4_keep_falling(cells):
    assert (r1(ring(cells, 40, 3, 0.1)["mean"]), r1(ring(cells, 320, 3, 0.1)["mean"])) == (43.1, 32.0)
    assert (r1(ring(cells, 40, 4, 0.1)["mean"]), r1(ring(cells, 320, 4, 0.1)["mean"])) == (254.7, 89.5)
    assert (round(ring(cells, 40, 4, 0.05)["mean"]), round(ring(cells, 320, 4, 0.05)["mean"])) == (3064, 454)
    for k in (3, 4):
        means = [ring(cells, n, k, 0.1)["mean"] for n in (40, 80, 160, 320)]
        assert means == sorted(means, reverse=True)


def test_h2b_small_n_ring_with_k4_is_slower_than_the_complete_graph(cells):
    for eps, complete, slow in ((0.2, 50.2, 63.5), (0.1, 495.9, 677.3)):
        assert r1(10 ** X.complete_log10(10, 0.4, eps)) == complete and r1(ring(cells, 10, 4, eps)["mean"]) == slow and slow > complete


# --- H3, H3b, H4 -----------------------------------------------------------------------------------------------------------------------------
def test_h3_slope_equals_the_chernoff_rate():
    rows = X.e2_rate_rows(0.4)
    assert [round(r["measured"], 4) for r in rows] == [0.1350, 0.2413, 0.3549, 0.5101]
    assert [round(r["chernoff"], 4) for r in rows] == [0.1352, 0.2415, 0.3551, 0.5103]
    assert all(round(-r["diff"], 4) == 0.0002 for r in rows)


def test_h3_other_q_values():
    got = [X.e2_rate_rows(q, (0.1,), 400, 800)[0] for q in (0.1, 0.2, 0.3, 0.4, 0.45)]
    assert [round(g["measured"], 4) for g in got] == [0.0086, 0.0603, 0.1402, 0.2411, 0.2985]
    assert [round(g["chernoff"], 4) for g in got] == [0.0090, 0.0607, 0.1406, 0.2415, 0.2989]


def test_h3b_rough_expectation_is_too_large():
    assert [round(r["rough_over_measured"], 2) for r in X.e2_rate_rows(0.4)] == [2.07, 1.66, 1.47, 1.33]


def test_h4_exponent_in_eps_is_close_to_thr():
    rows = X.e2_exponent_rows(0.4)
    assert [r["thr"] for r in rows] == [4, 8, 16, 32] and [round(r["a"], 2) for r in rows] == [3.67, 7.45, 15.01, 30.11]
    assert [round(r["a_over_thr"], 2) for r in rows] == [0.92, 0.93, 0.94, 0.94]


# --- H5 --------------------------------------------------------------------------------------------------------------------------------------
def test_h5_linear_system_breaks_where_stated_as_bands():
    """Die README nennt 3,0·10⁻⁹ / 5,3·10⁻⁷ / 1,0·10⁻⁴ / 1,9·10⁻² (Windows, numpy-Löser). Auf anderen Plattformen wandern die letzten Stellen: geprüft werden Bänder um eine Größenordnung
    (n = 30 bis 50) bzw. um den Faktor 3 (n = 60) und die Aussage zu n = 70 (negativ oder Abweichung über 50 %)."""
    rows = {r["n"]: r for r in X.e7_rows()}
    assert 3e-10 < rows[30]["deviation"] < 3e-8 and 5e-8 < rows[40]["deviation"] < 5e-6 and 1e-5 < rows[50]["deviation"] < 1e-3 and 6e-3 < rows[60]["deviation"] < 6e-2
    assert rows[70]["deviation"] > 0.5 and rows[70]["negative"] is True
    # Der Logarithmus ist plattformunabhängig; der Wert des linearen Systems nicht (Windows −7,0·10^15, Linux der CI −1,4·10^16): Vorzeichen und Größenordnung.
    assert rows[70]["log_time"] == pytest.approx(5.6e16, rel=0.01)
    assert rows[70]["linear"] < 0 and 1e15 < abs(rows[70]["linear"]) < 1e17


def test_h5_function_refuses_beyond_forty():
    with pytest.raises(ValueError):
        E.complete_linear_time(41, 17, 0.1)


# --- H6 --------------------------------------------------------------------------------------------------------------------------------------
def test_h6_advantage_ends_only_near_full_sight(cells):
    rows20, rows30, rows40 = X.e3_rows(cells, 20, 0.2), X.e3_rows(cells, 30, 0.2), X.e3_rows(cells, 40, 0.2)
    end20, end30 = V.first_advantage_end(rows20), V.first_advantage_end(rows30)
    assert (end20["k"], round(end20["share"], 2), round(end20["ratio"], 2)) == (8, 0.84, 1.78)
    assert (end30["k"], round(end30["share"], 2), round(end30["ratio"], 2)) == (13, 0.90, 1.94)
    assert all(r["ratio"] > 2 for r in rows20[:7]) and all(r["ratio"] > 2 for r in rows30[:12])
    assert (round(rows20[-1]["share"], 2), round(rows20[-1]["ratio"], 2)) == (0.95, 0.74) and (round(rows30[-1]["share"], 2), round(rows30[-1]["ratio"], 2)) == (0.97, 0.66)
    assert (r1(rows20[4]["ratio"]), round(rows20[4]["share"], 2)) == (42.3, 0.53) and (r1(rows30[7]["ratio"]), round(rows30[7]["share"], 2)) == (43.7, 0.55)
    assert (round(rows30[5]["share"], 2), round(rows30[5]["ratio"])) == (0.41, 421) and (round(rows40[7]["share"], 2), round(rows40[7]["ratio"])) == (0.41, 1039)
    assert all(r["capped"] == 0 for r in rows20 + rows30 + rows40)


def test_h6_complete_graph_values_used_for_the_ratio(cells):
    assert round(math.exp(E.complete_log_time(30, M.threshold(0.4, 29), 0.2))) == round(X.e3_rows(cells, 30, 0.2)[0]["complete"])


# --- H7 --------------------------------------------------------------------------------------------------------------------------------------
def test_h7_time_depends_on_thr_and_k_and_falls_with_the_spread(cells):
    rows = X.e4_rows(cells)
    by = {(r["k"], r["thr"]): r["mean"] for r in rows}
    assert [r1(by[(k, 4)]) for k in (4, 5, 6)] == [36.5, 15.8, 9.2]
    assert [r1(by[(k, k)]) for k in (3, 4, 5, 6)] == [14.2, 36.5, 116.9, 362.4]
    assert [round(f, 1) for _, _, f in V.spread_zero_factors(rows)] == [2.6, 3.2, 3.1]
    for a in rows:
        for b in rows:
            if (a["k"], a["thr"]) == (b["k"], b["thr"]):
                assert a["mean"] == b["mean"]


# --- H8, H8b ---------------------------------------------------------------------------------------------------------------------------------
def test_h8_grid_is_flat_in_L_from_twelve(cells):
    flat = [f for f in V.grid_flatness(X.e5_rows(cells)) if f["thr"] <= 3]
    assert len(flat) == 6 and all(f["capped"] == 0 for f in flat)
    assert r1(max(abs(f["change"]) for f in flat)) == 6.6
    rows = {(r["kind"], r["thr"], r["eps"], r["side"]): r["mean"] for r in X.e5_rows(cells)}
    assert (r1(rows[("vn", 2, 0.2, 4)]), r1(rows[("vn", 2, 0.2, 24)])) == (10.1, 7.7)


def test_h8b_moore_with_threshold_four_is_not_saturated(cells):
    rows = [r for r in X.e5_rows(cells) if r["kind"] == "moore" and r["thr"] == 4]
    low = {r["side"]: r for r in rows if r["eps"] == 0.1}
    high = {r["side"]: r for r in rows if r["eps"] == 0.2}
    assert [round(low[s]["mean"]) for s in (4, 8, 12, 24)] == [1706, 12126, 5162, 1228] and [low[s]["capped"] for s in (4, 8, 12, 24)] == [0, 1, 1, 3] and low[4]["total"] == 240
    assert [round(high[s]["mean"]) for s in (4, 8, 12, 24)] == [88, 82, 48, 36] and all(high[s]["capped"] == 0 for s in high)
    assert list(high[s]["mean"] for s in (4, 8, 12, 24)) == sorted((high[s]["mean"] for s in (4, 8, 12, 24)), reverse=True)


# --- H9 --------------------------------------------------------------------------------------------------------------------------------------
def test_h9_back_exponent_is_about_twice():
    rows = X.e6_rows(0.4)
    assert {r["eps"]: round(r["ratio"], 2) for r in rows if r["n"] == 640} == {0.2: 2.42, 0.1: 2.06, 0.05: 1.90}
    pair = X.e6_rate_pair(0.4, 0.1, 320, 640)
    assert (round(pair["slope_back"], 4), round(pair["rate_back"], 4), round(pair["slope_hin"], 4), round(pair["rate_hin"], 4)) == (0.4968, 0.4972, 0.2410, 0.2415)


# --- H10 -------------------------------------------------------------------------------------------------------------------------------------
def test_h10_nucleation_exponent_is_only_a_direction(cells):
    exps = []
    for k in (1, 2, 3, 4):
        a, b = ring(cells, 320, k, 0.1)["mean"], ring(cells, 320, k, 0.05)["mean"]
        exps.append(math.log(b / a) / math.log(2))
    assert [round(e, 2) for e in exps] == [0.49, 0.86, 1.44, 2.34]
    assert (r1(ring(cells, 160, 4, 0.1)["mean"]), r1(ring(cells, 320, 4, 0.1)["mean"])) == (103.1, 89.5)


# --- Beschreibung und Quellen ---------------------------------------------------------------------------------------------------------------
def test_description_numbers_of_the_model_section():
    assert 0.28 * 25 > 7.0 and M.threshold(0.28, 25) == 7
    assert E.EXACT_MAX_N == 12 and E.LINEAR_SAFE_MAX_N == 40 and GRID.SEEDS == (100001, 100002, 100003, 100004)


def test_spread_quantiles_of_the_ring_in_the_limits_section(cells):
    c80, c20 = ring(cells, 80, 2, 0.1), ring(cells, 20, 2, 0.1)
    assert (r1(c80["p10"]), r1(c80["median"]), r1(c80["p90"]), r1(c80["mean"]), c80["total"]) == (8.0, 12.0, 18.3, 12.7, 240)
    assert (r1(c20["p10"]), r1(c20["median"]), r1(c20["p90"]), r1(c20["mean"])) == (5.0, 12.0, 29.5, 14.8)


def test_ring_values_in_the_corrections_section(cells):
    assert [r1(ring(cells, n, 2, 0.1)["mean"]) for n in (10, 20, 40, 80, 160, 320)] == [21.5, 14.8, 12.0, 12.7, 12.5, 12.4]
    assert round(15.9 / ring(cells, 20, 2, 0.1)["mean"] - 1, 2) == 0.08


def test_sources_paragraph_names_only_the_checked_sources():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    quellen = text.split("## Quellen")[1].split("## Verifikation")[0]
    for needed in ("Kandori, Mailath, Rob (1993)", "Ellison (1993)", "Ellison (2000)", "Nicht geprüft und nicht verwendet: Young (1993), Blume (1993)"):
        assert needed in quellen
    assert "Young" not in text.split("Nicht geprüft und nicht verwendet")[0]


def test_logit_statement_of_the_limits_section_matches_the_sibling_readme():
    """„60 von 63 Zellen“: so steht es in der README von gleichgewichtsauswahl-demo (H8 dort)."""
    sibling = ROOT.parent / "gleichgewichtsauswahl-demo" / "README.md"
    if not sibling.exists():
        pytest.skip("gleichgewichtsauswahl-demo liegt nicht im Portfolio-Ordner")
    assert "60 von 63" in sibling.read_text(encoding="utf-8")
    assert "60 von 63 Zellen" in (ROOT / "README.md").read_text(encoding="utf-8") and "60 von 63 Zellen" in (ROOT / "app.py").read_text(encoding="utf-8")


def test_app_limits_text_and_readme_agree_on_the_assumptions():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("Lernregel und Eigennutz sind Modellannahmen", "Zeit nur als Erwartungswert", "nicht gesättigt"):
        assert phrase in app
    assert "Eigennutz und die Lernregel sind Modellannahmen" in readme and "Zeiten sind Erwartungswerte" in readme and "Nicht gesättigt" in readme
