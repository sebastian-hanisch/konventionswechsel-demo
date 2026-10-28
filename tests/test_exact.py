"""Exakte Zeiten: Einheiten (Raten, Zeit, lineares System, Netze), Zweig-Tests und Fehlerfälle."""

import math

import numpy as np
import pytest

import kvw_exact as E
import kvw_graphs as GR
import kvw_model as M


def test_birth_death_rates_by_hand(mini3):
    """n = 3, thr = 1, ε = 0,2: up = (ε/2, (2/3)(1 − ε/2), (1/3)(1 − ε/2)·0 …) und down = (0, (1/3)(1 − ε/2), (2/3)(ε/2 + (1 − ε)·[k − 1 < 1]), …)."""
    up, down = E.birth_death_rates(3, 1, 0.2)
    assert up[0] == pytest.approx(0.1) and up[1] == pytest.approx(2 / 3 * 0.9) and up[2] == pytest.approx(1 / 3 * 0.9) and up[3] == 0.0
    assert down[0] == 0.0 and down[1] == pytest.approx(1 / 3 * 0.9) and down[2] == pytest.approx(2 / 3 * (0.8 * 0 + 0.1)) and down[3] == pytest.approx(0.1)


def test_rates_rejects_bad_input():
    for n, eps in ((1, 0.1), (5, 0.0), (5, 1.0)):
        with pytest.raises(ValueError):
            E.birth_death_rates(n, 2, eps)


def test_eps_lowers_the_rate_strictly_so_time_falls_with_eps():
    """Zweig-Test: mehr Fehler = schnellerer Wechsel, in jedem Schritt der Leiter (Vollgraph n = 20, thr 8)."""
    times = [E.complete_log_time(20, 8, eps) for eps in (0.02, 0.05, 0.1, 0.2, 0.3)]
    assert all(a > b for a, b in zip(times, times[1:]))


def test_time_depends_on_thr_not_on_q_directly():
    """q = 0,41 und q = 0,43 geben bei d = 19 beide thr = 8 (7,79 und 8,17: ⌈7,79⌉ = 8, ⌈8,17⌉ = 9): hier zwei q mit GLEICHEM thr, zwei mit verschiedenem."""
    d = 19
    same = [M.threshold(q, d) for q in (0.40, 0.41, 0.42)]
    assert same == [8, 8, 8]
    assert E.complete_log_time(20, same[0], 0.1) == E.complete_log_time(20, same[2], 0.1)
    assert M.threshold(0.43, d) == 9 and E.complete_log_time(20, 9, 0.1) > E.complete_log_time(20, 8, 0.1)


def test_target_defaults_to_the_half_and_a_larger_target_takes_longer():
    n, thr = 20, 8
    assert E.complete_log_time(n, thr, 0.1) == E.complete_log_time(n, thr, 0.1, target=10)
    assert E.complete_log_time(n, thr, 0.1, target=12) > E.complete_log_time(n, thr, 0.1, target=10) > E.complete_log_time(n, thr, 0.1, target=5)
    for bad in (0, 21):
        with pytest.raises(ValueError):
            E.complete_log_time(n, thr, 0.1, target=bad)


def test_log_time_is_finite_where_the_number_itself_does_not_fit():
    """n = 2 000: log10 T liegt über 300, die Zahl selbst überschreitet das, was ein float als exp(ln T) fasst (ln T > 709); der Logarithmus bleibt endlich."""
    value = E.complete_log10_time(2000, M.threshold(0.4, 1999), 0.1)
    assert math.isfinite(value) and value * math.log(10) > 709
    with pytest.raises(OverflowError):
        math.exp(value * math.log(10))


def test_linear_system_refuses_large_n_unless_unsafe():
    with pytest.raises(ValueError, match="numerisch unbrauchbar"):
        E.complete_linear_time(41, 17, 0.1)
    E.complete_linear_time(40, 16, 0.1)                       # Grenze inklusive
    assert np.isfinite(E.complete_linear_time(70, 28, 0.1, unsafe=True))


def test_linear_system_target_validation():
    for bad in (0, 8):
        with pytest.raises(ValueError):
            E.complete_linear_time(7, 3, 0.1, target=bad)
    assert E.complete_linear_time(7, 3, 0.1, target=4) == E.complete_linear_time(7, 3, 0.1)     # ⌈7/2⌉ = 4


def test_graph_exact_refuses_large_n():
    with pytest.raises(ValueError, match="Zustandsraum"):
        E.graph_exact_time(GR.ring_nb(13, 1), 1, 0.1)


def test_network_changes_the_time_against_the_neighbours():
    """Zweig-Test (jedes Netz weicht von den anderen ab), n = 9 (3 × 3 Torus), ε = 0,3, q = 0,4: Vollgraph, Ring k = 2, Torus vn und Torus moore (= Vollgraph bei 3 × 3)."""
    n, eps = 9, 0.3
    complete = E.graph_exact_time(GR.complete_nb(n), M.threshold(0.4, 8), eps)
    ring = E.graph_exact_time(GR.ring_nb(n, 2), M.threshold(0.4, 4), eps)
    vn = E.graph_exact_time(GR.torus_nb(3, 3, "vn"), M.threshold(0.4, 4), eps)
    assert len({round(complete, 6), round(ring, 6), round(vn, 6)}) == 3
    assert vn != ring and ring < complete


def test_chernoff_rate_and_kl():
    assert E.kl(0.4, 0.05) == pytest.approx(0.4 * math.log(8) + 0.6 * math.log(0.6 / 0.95))
    assert E.chernoff_rate(0.4, 0.1) == pytest.approx(E.kl(0.4, 0.05) / math.log(10))
    assert E.kl(0.5, 0.5) == pytest.approx(0.0)
    for q, p in ((0.0, 0.1), (0.4, 1.0)):
        with pytest.raises(ValueError):
            E.kl(q, p)
    assert E.rough_rate(0.4, 0.1) == pytest.approx(0.4)


def test_mirror_values_and_symmetry_of_the_game():
    """n = 20, q = 0,4: thr = 8; die gespiegelte Kette braucht thr_B = 12 und das Ziel 14; die Zeit zurück ist länger als hin."""
    assert E.mirror(20, 8) == (12, 14)
    assert E.mirror_log_time(20, 8, 0.1) == E.complete_log_time(20, 12, 0.1, 14) > E.complete_log_time(20, 8, 0.1)
