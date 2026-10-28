"""Orakel für den Verlauf eines Laufs, die Zusammenfassung von Läufen und die Auswertung der Messzellen: jeweils ein anderer Rechenweg.

  * `kvw_trace`: der Zählerverlauf eines Laufs gegen eine wörtliche Schleife mit dem skalaren SplitMix64-Strom (Vollgraph über k, Ring über die ganze Belegung);
  * `kvw_evaluation.summarise`: Mittel, Standardfehler, Median und Quantile gegen `statistics` und numpy;
  * `kvw_grid.measure_cell`: Mittel und Zahlen aus einer Neuberechnung mit einer Schleife über die einzelnen Läufe (`simulate_one`) statt der vektorisierten Fassung."""

import math
import statistics

import numpy as np
import pytest

import kvw_evaluation as V
import kvw_graphs as GR
import kvw_grid as GRID
import kvw_model as M
import kvw_rng as R
import kvw_sim as SI
import kvw_trace as TR


def literal_ring_series(nb, thr, eps, stream, cap):
    rng = R.SplitMix64(stream)
    n = len(nb)
    state = [0] * n
    series = []
    for _ in range(cap):
        u0, u1, u2 = rng.uniform(), rng.uniform(), rng.uniform()
        i = int(u0 * n)
        new = (1 if u2 < 0.5 else 0) if u1 < eps else (1 if sum(state[j] for j in nb[i]) >= thr else 0)
        state[i] = new
        series.append(sum(state))
        if series[-1] >= (n + 1) // 2:
            break
    return series


def literal_complete_series(n, thr, eps, stream, cap, target):
    rng = R.SplitMix64(stream)
    k, series = 0, []
    for _ in range(cap):
        u0, u1, u2 = rng.uniform(), rng.uniform(), rng.uniform()
        is_a = u0 * n < k
        seen = k - 1 if is_a else k
        new = (1 if u2 < 0.5 else 0) if u1 < eps else (1 if seen >= thr else 0)
        k += new - (1 if is_a else 0)
        series.append(k)
        if k >= target:
            break
    return series


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trace_series_on_a_ring_equals_the_literal_loop(seed):
    nb = GR.ring_nb(14, 2)
    thr = M.threshold(0.4, 4)
    tr = TR.trace_run("ring", 14, 2, thr, 0.2, seed, 50000)
    assert tr.series == literal_ring_series(nb.tolist(), thr, 0.2, SI.stream_seed(seed, 0), 50000)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trace_series_on_the_complete_graph_equals_the_literal_loop(seed):
    thr = M.threshold(0.4, 11)
    tr = TR.trace_run("complete", 12, None, thr, 0.3, seed, 50000)
    assert tr.series == literal_complete_series(12, thr, 0.3, SI.stream_seed(seed, 0), 50000, 6)


def test_trace_series_of_the_back_direction_equals_the_literal_loop():
    thr_back, target_back = 6, 8                     # n = 10, thr = 4
    tr = TR.trace_run("complete", 10, None, thr_back, 0.3, 5, 100000, target_back)
    assert tr.series == literal_complete_series(10, thr_back, 0.3, SI.stream_seed(5, 0), 100000, target_back)


def test_summarise_equals_statistics_and_numpy():
    rng = np.random.default_rng(0)                   # nur Testdaten, nicht der Zufall der Demo
    times = np.round(rng.exponential(10.0, size=57) * 20) / 20
    times[[3, 11]] = np.nan
    sm = V.summarise(times, 1000, 20, exact=9.0)
    fin = [t for t in times if t == t]
    assert sm.finished == len(fin) == 55 and sm.capped == 2 and sm.runs == 57
    assert sm.mean == pytest.approx(statistics.fmean(fin)) and sm.se == pytest.approx(statistics.stdev(fin) / math.sqrt(len(fin)))
    assert sm.median == pytest.approx(statistics.median(fin)) and sm.p10 == pytest.approx(np.quantile(fin, 0.1)) and sm.p90 == pytest.approx(np.quantile(fin, 0.9))
    assert sm.z == pytest.approx((statistics.fmean(fin) - 9.0) / sm.se) and sm.cap_time == 50.0


def test_measure_cell_equals_a_loop_over_single_runs():
    """Eine Zelle (Ring n = 12, k = 2, thr 2, ε = 0,3, 4 × 6 Läufe) aus `simulate_one` zusammengesetzt, ohne die vektorisierte Fassung."""
    spec = {"kind": "ring", "n": 12, "k": 2, "thr": 2, "eps": 0.3, "runs": 6, "cap": 100000, "experiments": ["e1"]}
    cell = GRID.measure_cell(spec)
    nb = GR.ring_nb(12, 2)
    pooled = []
    for seed in GRID.SEEDS:
        for run in range(6):
            t, _ = SI.simulate_one(nb, 2, 0.3, SI.stream_seed(seed, run), 100000)
            pooled.append(t)
    assert cell["finished"] == len(pooled) and cell["capped"] == 0 and cell["total"] == 24
    assert cell["mean"] == pytest.approx(statistics.fmean(pooled)) and cell["se"] == pytest.approx(statistics.stdev(pooled) / math.sqrt(24))
    assert cell["median"] == pytest.approx(statistics.median(pooled)) and cell["p10"] == pytest.approx(np.percentile(pooled, 10))
