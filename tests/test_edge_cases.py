"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Instanzen, Gleichstände, inklusive Grenzen, Reglerränder."""

import math

import pytest

import kvw_constants as C
import kvw_evaluation as V
import kvw_exact as E
import kvw_graphs as GR
import kvw_model as M
import kvw_scenario as S
import kvw_sim as SI


def test_two_firms_complete_graph():
    """n = 2: Ziel 1 (⌈2/2⌉); aus „alle B“ führt jede Wahl mit A zum Ziel: Schritte = 1/(ε/2), je Betrieb /2."""
    eps = 0.2
    thr = M.threshold(0.4, 1)
    assert thr == 1
    assert math.exp(E.complete_log_time(2, thr, eps)) == pytest.approx((2 / eps) / 2)
    assert E.graph_exact_time(GR.complete_nb(2), thr, eps) == pytest.approx((2 / eps) / 2)


def test_smallest_slider_values_run_through_the_kernel():
    s = V.Settings(network=C.NETWORKS[1], n=C.N_MIN, k=C.K_MAX, q=C.Q_MAX, eps=C.EPS_MIN, budget=C.SIM_BUDGETS[0], runs=C.RUNS_MIN)
    view = V.analyse_main(s)
    assert view.spec.n == 4 and view.spec.k == 1 and view.spec.thr == 1
    sm = V.run_main_simulation(view)
    assert sm.finished + sm.capped == C.RUNS_MIN


def test_largest_slider_values_run_through_the_kernel():
    for network in C.NETWORKS:
        s = V.Settings(network=network, n=C.N_MAX, k=C.K_MAX, q=C.Q_MIN, eps=C.EPS_MAX, budget=C.SIM_BUDGETS[0], runs=C.RUNS_MIN)
        view = V.analyse_main(s)
        sm = V.run_main_simulation(view)
        assert sm.finished + sm.capped == C.RUNS_MIN and view.spec.n <= C.N_MAX


def test_complete_graph_at_the_largest_n_and_smallest_eps_is_finite_in_the_log():
    view = V.analyse_main(V.Settings(n=C.N_MAX, q=C.Q_MAX, eps=C.EPS_MIN))
    assert math.isfinite(view.log10_time) and view.log10_time > 100 and view.exceeds_budget


def test_threshold_exactly_at_half_of_the_partners_is_inclusive():
    """q·d ganzzahlig (q = 0,25, d = 4): thr = 1, nicht 2; 0,5 ist nicht erlaubt (q < 1)."""
    assert M.threshold(0.25, 4) == 1 and M.threshold(0.5, 4) == 2 and M.threshold(0.5, 5) == 3


def test_ring_with_n_equal_two_k_plus_one_is_the_complete_graph_in_time():
    nb_ring = GR.ring_nb(7, 3)
    thr = M.threshold(0.4, 6)
    assert E.graph_exact_time(nb_ring, thr, 0.3) == pytest.approx(math.exp(E.complete_log_time(7, thr, 0.3)), rel=1e-9)


def test_grid_with_eight_neighbours_on_a_three_by_three_torus_is_the_complete_graph():
    nb = GR.torus_nb(3, 3, "moore")
    thr = M.threshold(0.4, 8)
    assert E.graph_exact_time(nb, thr, 0.3) == pytest.approx(math.exp(E.complete_log_time(9, thr, 0.3)), rel=1e-9)


def test_goal_for_odd_and_even_n():
    """n = 5: Ziel 3 (nicht 2,5 und nicht 2); n = 6: Ziel 3."""
    assert M.goal(5) == 3 and M.goal(6) == 3
    res = SI.simulate(GR.complete_nb(5), 2, 0.3, 30, 1, 100000)
    assert res.capped == 0 and (res.finished * 5 >= 3).all()


def test_back_direction_with_target_equal_to_n_is_capped_at_n():
    """Sehr hohe Schwelle: thr_B + 2 > n wird auf n begrenzt (kein Index außerhalb)."""
    thr = M.threshold(0.45, 3)                  # n = 4: thr = 2
    thr_back, target_back = E.mirror(4, thr)
    assert thr_back == 2 and target_back == 4
    assert math.isfinite(E.mirror_log_time(4, thr, 0.1))


def test_direction_back_is_ignored_for_other_networks_and_says_so():
    view = V.analyse_main(V.Settings(network=C.NETWORKS[1], n=40, direction=C.DIRECTIONS[1]))
    assert view.direction_ignored and not view.back and view.thr == view.spec.thr
    kind, text = V.verdict_main(view)
    assert kind == "info" and "Rückwechsel ist nur für den Vollgraph" in text


def test_k_is_ignored_outside_the_ring_and_limited_inside():
    assert S.resolve_network(C.NETWORKS[0], 20, 14, 0.4).k is None
    assert S.resolve_network(C.NETWORKS[1], 6, 14, 0.4).k == 2          # (6 − 1)//2


def test_all_runs_capped_gives_no_mean():
    sm = V.summarise(SI.simulate(GR.complete_nb(20), 8, 0.1, 6, 1, 30).times, 30, 20)
    assert sm.finished == 0 and sm.capped == 6 and math.isnan(sm.mean)
    kind, text = V.verdict_simulation(sm)
    assert kind == "warning" and "Kein Lauf von 6" in text and "ein Mittel wird nicht gebildet" in text
