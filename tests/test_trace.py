"""Der Verlauf eines Laufs: derselbe Strom wie Lauf 0 der Simulation, Bilder an den richtigen Schritten."""

import math

import pytest

import kvw_graphs as GR
import kvw_model as M
import kvw_sim as SI
import kvw_trace as TR


def test_trace_time_equals_run_zero_of_the_simulation_on_a_ring():
    nb = GR.ring_nb(20, 2)
    thr = M.threshold(0.4, 4)
    tr = TR.trace_run("ring", 20, 2, thr, 0.1, 35, 100000)
    assert tr.reached and tr.time == SI.simulate(nb, thr, 0.1, 3, 35, 100000).times[0]
    assert len(tr.series) == round(tr.time * 20) and tr.series[-1] >= tr.target == 10


def test_trace_time_equals_run_zero_on_the_complete_graph():
    thr = M.threshold(0.4, 11)
    tr = TR.trace_run("complete", 12, None, thr, 0.3, 35, 100000)
    assert tr.reached and tr.time == SI.simulate_complete(12, thr, 0.3, 3, 35, 100000).times[0]


def test_trace_of_the_back_direction_uses_the_mirrored_target():
    n = 10
    thr = M.threshold(0.4, n - 1)
    from kvw_exact import mirror
    thr_back, target_back = mirror(n, thr)
    tr = TR.trace_run("complete", n, None, thr_back, 0.3, 4, 200000, target_back)
    assert tr.reached and tr.target == target_back and tr.series[-1] >= target_back
    assert tr.time == SI.simulate_complete(n, thr_back, 0.3, 2, 4, 200000, target_back).times[0]


def test_trace_reports_a_miss_at_the_cap():
    tr = TR.trace_run("complete", 20, None, 8, 0.1, 35, 300)
    assert not tr.reached and math.isnan(tr.time) and len(tr.series) == 300


def test_trace_on_a_grid():
    tr = TR.trace_run("vn", 36, None, M.threshold(0.25, 4), 0.2, 3, 100000)
    assert tr.reached and tr.series[-1] >= 18


def test_frame_steps_include_both_ends_ascending_and_unique():
    steps = TR.frame_steps(1000, 40)
    assert steps[0] == 0 and steps[-1] == 1000 and steps == sorted(set(steps)) and len(steps) == 40
    assert TR.frame_steps(3, 40) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        TR.frame_steps(10, 1)
    with pytest.raises(ValueError):
        TR.frame_steps(-1, 5)


def test_frames_have_the_count_of_the_series_at_those_steps():
    thr = M.threshold(0.4, 4)
    tr = TR.trace_run("ring", 20, 2, thr, 0.1, 35, 100000)
    steps = TR.frame_steps(len(tr.series), 12)
    frames = TR.frames("ring", 20, 2, thr, 0.1, 35, steps)
    assert [sum(f) for f in frames] == [0 if s == 0 else tr.series[s - 1] for s in steps]
    assert all(len(f) == 20 for f in frames)


def test_complete_graph_has_no_picture():
    with pytest.raises(ValueError):
        TR.frames("complete", 10, None, 4, 0.1, 1, [0, 1])
