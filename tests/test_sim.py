"""Simulation: Strom-Layout (SplitMix64), vektorisierte gegen einfache Fassung, Abbruch an der Schrittgrenze, Auswertung."""

import math

import numpy as np
import pytest

import kvw_graphs as GR
import kvw_model as M
import kvw_rng as R
import kvw_sim as SI


def literal_run(nb, thr, eps, stream, cap):
    """Das Verfahren Wort für Wort mit dem skalaren SplitMix64-Strom (drei Zahlen je Schritt) – unabhängig von uniform_block und von den Blöcken."""
    rng = R.SplitMix64(stream)
    n = len(nb)
    target = (n + 1) // 2
    state = [0] * n
    for step in range(1, cap + 1):
        u0, u1, u2 = rng.uniform(), rng.uniform(), rng.uniform()
        i = int(u0 * n)
        if u1 < eps:
            new = 1 if u2 < 0.5 else 0
        else:
            new = 1 if sum(state[j] for j in nb[i]) >= thr else 0
        state[i] = new
        if sum(state) >= target:
            return step / n
    return float("nan")


@pytest.mark.parametrize("name,nb,eps", [("Ring 12, k = 2", GR.ring_nb(12, 2), 0.2), ("Torus 3×4 vn", GR.torus_nb(3, 4, "vn"), 0.3), ("Vollgraph 9", GR.complete_nb(9), 0.3)])
def test_batch_equals_the_literal_loop_run_by_run(name, nb, eps):
    thr = M.threshold(0.4, nb.shape[1])
    res = SI.simulate(nb, thr, eps, 6, 77, 20000)
    for run in range(6):
        want = literal_run(nb.tolist(), thr, eps, SI.stream_seed(77, run), 20000)
        assert res.times[run] == pytest.approx(want, nan_ok=True), (name, run)


def test_streams_continue_across_the_block_boundary():
    """Läufe, die länger als ein Block (2 048 Schritte) dauern, stimmen mit der wörtlichen Schleife überein: der Strom läuft über die Blockgrenze weiter."""
    nb = GR.ring_nb(40, 3)
    thr = M.threshold(0.4, 6)
    res = SI.simulate(nb, thr, 0.05, 4, 5, 60000)
    assert (res.times * 40 > SI.CHUNK).sum() >= 2
    for run in range(4):
        assert res.times[run] == pytest.approx(literal_run(nb.tolist(), thr, 0.05, SI.stream_seed(5, run), 60000), nan_ok=True)


def test_a_run_does_not_depend_on_how_many_runs_share_the_batch():
    nb = GR.ring_nb(20, 2)
    thr = M.threshold(0.4, 4)
    small = SI.simulate(nb, thr, 0.1, 3, 9, 50000).times
    large = SI.simulate(nb, thr, 0.1, 9, 9, 50000).times
    assert np.array_equal(small, large[:3])


def test_same_seed_same_numbers_different_seed_different():
    nb = GR.ring_nb(20, 2)
    thr = M.threshold(0.4, 4)
    a, b, c = (SI.simulate(nb, thr, 0.1, 8, s, 50000).times for s in (3, 3, 4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_scalar_reference_run_equals_batch_run_zero_and_returns_the_series():
    nb = GR.ring_nb(15, 2)
    thr = M.threshold(0.4, 4)
    t, series = SI.simulate_one(nb, thr, 0.2, SI.stream_seed(5, 0), 100000)
    batch = SI.simulate(nb, thr, 0.2, 1, 5, 100000).times[0]
    assert t == batch and len(series) == round(t * 15) and series[-1] >= 8 and all(abs(b - a) <= 1 for a, b in zip(series, series[1:]))


def test_cap_reports_instead_of_averaging():
    """Schrittgrenze 50 bei n = 20 und ε = 0,1 im Vollgraph (Zeit 10^4,95 je Betrieb): kein Lauf kann das Ziel erreichen. Die Zeiten sind nan, die Zahl der abgebrochenen Läufe stimmt, kein Mittel."""
    thr = M.threshold(0.4, 19)
    res = SI.simulate(GR.complete_nb(20), thr, 0.1, 5, 1, 50)
    assert res.capped == 5 and np.isnan(res.times).all() and res.finished.size == 0 and res.cap_steps == 50
    mean, se = SI.mean_and_error(res.times)
    assert math.isnan(mean) and math.isnan(se)
    lumped = SI.simulate_complete(20, thr, 0.1, 5, 1, 50)
    assert lumped.capped == 5


def test_a_mixed_batch_counts_only_finished_runs_in_the_mean():
    nb = GR.ring_nb(20, 3)
    thr = M.threshold(0.4, 6)
    res = SI.simulate(nb, thr, 0.1, 40, 17, 1500)           # Grenze zwischen kürzester und längster Zeit
    assert 0 < res.capped < 40
    assert res.capped == int(np.isnan(res.times).sum()) and len(res.finished) == 40 - res.capped
    mean, _ = SI.mean_and_error(res.times)
    assert mean == pytest.approx(float(res.finished.mean())) and mean <= 1500 / 20


def test_times_are_multiples_of_one_over_n_and_start_is_all_b():
    nb = GR.ring_nb(12, 2)
    res = SI.simulate(nb, M.threshold(0.4, 4), 0.3, 20, 2, 50000)
    assert all(abs(t * 12 - round(t * 12)) < 1e-9 for t in res.finished)
    assert (res.finished * 12 >= 6).all()               # mindestens 6 Schritte (je Schritt höchstens ein neuer A-Betrieb)


def test_complete_lumped_equals_literal_loop_on_the_counted_chain():
    """Gezählte Fassung: dieselbe Zahl der A-Betriebe wie die wörtliche Schleife, die nur k verfolgt."""
    n, thr, eps = 12, 5, 0.3
    res = SI.simulate_complete(n, thr, eps, 5, 3, 100000)
    for run in range(5):
        rng = R.SplitMix64(SI.stream_seed(3, run))
        k, want = 0, float("nan")
        for step in range(1, 100001):
            u0, u1, u2 = rng.uniform(), rng.uniform(), rng.uniform()
            is_a = u0 * n < k
            seen = k - 1 if is_a else k
            new = (1 if u2 < 0.5 else 0) if u1 < eps else (1 if seen >= thr else 0)
            k += new - (1 if is_a else 0)
            if k >= 6:
                want = step / n
                break
        assert res.times[run] == pytest.approx(want, nan_ok=True)


def test_eps_makes_the_change_faster_on_average():
    nb = GR.ring_nb(20, 2)
    thr = M.threshold(0.4, 4)
    means = [SI.mean_and_error(SI.simulate(nb, thr, eps, 80, 6, 200000).times)[0] for eps in (0.05, 0.1, 0.3)]
    assert means[0] > means[1] > means[2]


def test_mean_and_error():
    assert SI.mean_and_error(np.array([1.0, 3.0, np.nan])) == (2.0, pytest.approx(1.0))
    m, se = SI.mean_and_error(np.array([5.0]))
    assert m == 5.0 and math.isnan(se)


def test_snapshots_follow_the_same_stream():
    nb = GR.ring_nb(12, 2)
    thr = M.threshold(0.4, 4)
    t, series = SI.simulate_one(nb, thr, 0.3, SI.stream_seed(8, 0), 50000)
    steps = [0, 5, len(series) // 2, len(series)]
    frames = SI.snapshots(nb, thr, 0.3, SI.stream_seed(8, 0), steps)
    assert [sum(f) for f in frames] == [0, series[4], series[len(series) // 2 - 1], series[-1]]
    assert SI.snapshots(nb, thr, 0.3, SI.stream_seed(8, 0), []) == []
    assert SI.snapshots(nb, thr, 0.3, SI.stream_seed(8, 0), [3, 3]) == [frames_at for frames_at in SI.snapshots(nb, thr, 0.3, SI.stream_seed(8, 0), [3])]


def test_stream_seed_is_injective_for_distinct_inputs():
    seeds = {SI.stream_seed(s, r) for s in range(0, 1000, 7) for r in range(200)}
    assert len(seeds) == len(range(0, 1000, 7)) * 200
