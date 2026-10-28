"""Randfälle der exakten Rechnung und der Auswertung: Längen, inklusive Grenzen, Gleichstände und die Meldung der Fehler (nicht nur irgendein ValueError)."""

import math

import pytest

import kvw_exact as E


def test_rates_have_one_entry_per_state_including_zero_and_n():
    """k = 0 .. n: n + 1 Einträge (ein Lauf über n + 2 würde unbemerkt einen negativen Aufwärtswert für k = n + 1 führen)."""
    up, down = E.birth_death_rates(7, 3, 0.2)
    assert len(up) == 8 and len(down) == 8 and up[7] == 0.0


@pytest.mark.parametrize("target", [1, 2, 7])
def test_linear_system_accepts_every_target_from_one_to_n(target):
    """Grenzen inklusive: target = 1 und target = n sind gültig (n = 7)."""
    assert E.complete_linear_time(7, 3, 0.2, target=target) > 0
    assert E.complete_log_time(7, 3, 0.2, target=target) == pytest.approx(math.log(E.complete_linear_time(7, 3, 0.2, target=target)), rel=1e-9)


@pytest.mark.parametrize("target", [0, 8, -1])
def test_targets_outside_one_to_n_are_rejected_with_the_message(target):
    with pytest.raises(ValueError, match="liegt außerhalb von 1 bis 7"):
        E.complete_log_time(7, 3, 0.2, target=target)
    with pytest.raises(ValueError, match="liegt außerhalb von 1 bis 7"):
        E.complete_linear_time(7, 3, 0.2, target=target)


@pytest.mark.parametrize("q,p", [(0.0, 0.1), (1.0, 0.1), (0.4, 0.0), (0.4, 1.0), (-0.1, 0.1), (0.4, 1.5)])
def test_kl_rejects_arguments_on_or_outside_the_boundary_with_the_message(q, p):
    with pytest.raises(ValueError, match="müssen in"):
        E.kl(q, p)


# --- Randfälle der Auswertung -------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402

import kvw_constants as C  # noqa: E402
import kvw_evaluation as V  # noqa: E402
import kvw_scenario as S  # noqa: E402


def test_exact_small_is_only_for_non_complete_networks_up_to_the_limit():
    ring12 = S.resolve_network(C.NETWORKS[1], 12, 2, 0.4)
    ring13 = S.resolve_network(C.NETWORKS[1], 13, 2, 0.4)
    complete = S.resolve_network(C.NETWORKS[0], 10, 2, 0.4)
    assert V.exact_small_log10(ring12, 0.3) is not None and V.exact_small_log10(ring13, 0.3) is None and V.exact_small_log10(complete, 0.3) is None


def test_exceeds_budget_is_strict_at_equality():
    spec = S.resolve_network(C.NETWORKS[0], 20, 2, 0.4)
    equal = V.MainView(V.Settings(), spec, False, False, 8, 10, 2.0, 100.0)             # log10(100) = 2: gleich der Grenze
    assert equal.exceeds_budget is False
    assert V.MainView(V.Settings(), spec, False, False, 8, 10, 2.0001, 100.0).exceeds_budget is True
    assert V.MainView(V.Settings(), spec, False, False, 8, 10, None, 100.0).exceeds_budget is None


def test_advantage_end_includes_a_factor_of_exactly_two():
    rows = [{"k": 1, "ratio": 9.0}, {"k": 2, "ratio": 2.0}, {"k": 3, "ratio": 1.0}]
    assert V.first_advantage_end(rows)["k"] == 2


def test_sawtooth_needs_a_strict_fall():
    rows = [{"k": 1, "mean": 5.0}, {"k": 2, "mean": 5.0}, {"k": 3, "mean": 4.0}]
    assert V.sawtooth_steps(rows) == [(2, 3)]


def test_ring_flatness_with_exactly_two_cells_and_with_one():
    def row(n, mean):
        return {"n": n, "rings": {1: {"mean": mean, "capped": 0}}}
    assert V.ring_flatness([row(20, 4.0), row(40, 5.0)], 1, 20) == pytest.approx(25.0)
    assert V.ring_flatness([row(20, 4.0), row(40, 5.0)], 1, 40) is None
    assert V.ring_flatness([row(20, 4.0), row(40, float("nan"))], 1, 20) is None


def test_grid_flatness_needs_both_sides_with_values():
    def row(side, mean):
        return {"kind": "vn", "q": 0.25, "thr": 1, "edge": 1, "eps": 0.2, "side": side, "mean": mean, "se": 0.1, "capped": 0, "total": 240}
    assert V.grid_flatness([row(12, 2.0)]) == [] and V.grid_flatness([row(24, 2.0)]) == []
    assert V.grid_flatness([row(12, None), row(24, 2.0)]) == [] and V.grid_flatness([row(12, 2.0), row(24, None)]) == []
    assert len(V.grid_flatness([row(12, 2.0), row(24, 2.2)])) == 1


def test_main_simulation_gives_an_exact_value_only_below_log10_300():
    """n = 1 245: log10 T = 299,87 (exakter Wert da); n = 1 248: 300,37 (kein exakter Wert: die Zahl 10^300 wird nicht mehr als float geführt)."""
    for n, has_exact in ((1245, True), (1248, False)):
        view = V.analyse_main(V.Settings(n=n, runs=2, budget=1000))
        assert view.log10_time == pytest.approx(299.87 if has_exact else 300.37, abs=0.01)
        assert (V.run_main_simulation(view).exact is not None) == has_exact


def test_summarise_with_zero_spread_has_no_z():
    sm = V.summarise(np.array([3.0, 3.0, 3.0]), 100, 10, exact=3.0)
    assert sm.se == 0.0 and sm.z is None and sm.mean == 3.0
    assert V.summarise(np.array([3.0, 4.0]), 100, 10, exact=None).z is None


def e1_row(n, k1, k2, k3, k4):
    return {"n": n, "complete_log10": 2.0, "rings": {1: k1, 2: k2, 3: k3, 4: k4}}


def cell(mean):
    return {"mean": mean, "se": 0.1, "capped": 0, "total": 240}


def test_verdict_e1_needs_both_flat_rings_and_skips_missing_cells_of_the_falling_rings():
    full = [e1_row(n, cell(4.0), cell(10.0), cell(30.0), cell(90.0)) for n in (20, 40, 320)]
    only_k1 = [e1_row(n, cell(4.0), None, None, None) for n in (20, 40, 320)]
    only_k2 = [e1_row(n, None, cell(10.0), None, None) for n in (20, 40, 320)]
    assert "fehlen" in V.verdict_e1(only_k1, 0.1)[1] and "fehlen" in V.verdict_e1(only_k2, 0.1)[1]
    no_k3_at_40 = [e1_row(20, cell(4.0), cell(10.0), cell(30.0), cell(90.0)), e1_row(40, cell(4.0), cell(10.0), None, cell(90.0)), e1_row(320, cell(4.0), cell(10.0), cell(30.0), cell(60.0))]
    text = V.verdict_e1(no_k3_at_40, 0.1)[1]
    assert "Bei k = 3 fällt" not in text and "Bei k = 4 fällt die Zeit von n = 40 auf n = 320 noch auf das 0,67-fache" in text
    assert "Bei k = 3 fällt die Zeit" in V.verdict_e1(full, 0.1)[1]


def e3_row(k, ratio, share=0.5):
    return {"k": k, "d": 2 * k, "share": share, "thr": 2, "mean": 4.0, "se": 0.1, "capped": 0, "total": 160, "ratio": ratio, "complete": 100.0}


def test_verdict_e3_last_row_is_slower_only_below_one():
    for ratio, slower in ((0.99, True), (1.0, False), (1.5, False), (float("nan"), False)):
        text = V.verdict_e3([e3_row(1, 50.0), e3_row(2, ratio, 0.95)], 30, 0.2)[1]
        assert ("ist der Ring langsamer" in text) is slower, ratio


def e5_row(kind, q, thr, side, mean, capped):
    return {"kind": kind, "q": q, "thr": thr, "edge": 3, "eps": 0.1, "side": side, "mean": mean, "se": 0.1, "capped": capped, "total": 240}


def test_verdict_e5_separates_unsaturated_cells_by_threshold_or_aborted_runs():
    stable = [e5_row("moore", 0.3, 3, 12, 10.0, 0), e5_row("moore", 0.3, 3, 24, 9.5, 0)]
    assert V.verdict_e5(stable)[0] == "info"
    aborted_thr3 = [e5_row("moore", 0.3, 3, 12, 10.0, 1), e5_row("moore", 0.3, 3, 24, 9.5, 0)]
    kind, text = V.verdict_e5(aborted_thr3)
    assert kind == "warning" and "1 Läufe in den beiden größten Gittern" in text
    thr4_clean = [e5_row("moore", 0.4, 4, 12, 50.0, 0), e5_row("moore", 0.4, 4, 24, 40.0, 0)]
    assert V.verdict_e5(thr4_clean)[0] == "warning" and "In 0 Zellen" in V.verdict_e5(thr4_clean)[1]


def test_verdict_e7_one_percent_is_included():
    rows = [{"n": 10, "deviation": 1e-9, "negative": False}, {"n": 50, "deviation": 1e-2, "negative": False}]
    kind, text = V.verdict_e7(rows)
    assert kind == "warning" and "Ab n = 50 weicht es um mehr als 1 % ab." in text


def test_verdict_simulation_with_exact_but_without_z_is_plain_info():
    sm = V.SimSummary(runs=3, finished=3, capped=0, mean=5.0, se=0.0, median=5.0, p10=5.0, p90=5.0, cap_time=100.0, exact=5.0, z=None)
    kind, text = V.verdict_simulation(sm)
    assert kind == "info" and "Exakt erwartet" not in text
