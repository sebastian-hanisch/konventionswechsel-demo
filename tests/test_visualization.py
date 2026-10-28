"""Abbildungen: alle Achsen gesperrt, Inhalte (Farben, Achsentyp, Spuren) gegen die Daten."""

import math

import pytest

import kvw_constants as C
import kvw_experiments as X
import kvw_visualization as Z


def locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def fake_cell(mean, capped=0, se=0.5):
    return {"mean": mean, "se": se, "capped": capped}


def test_trace_figure_marks_the_goal_and_a_miss():
    reached = Z.build_trace([0, 1, 1, 2, 3], 6, 3, True)
    missed = Z.build_trace([0, 1, 0, 1], 6, 3, False)
    assert locked(reached) and locked(missed)
    assert list(reached.data[0].y) == pytest.approx([0, 1 / 6, 1 / 6, 2 / 6, 3 / 6]) and reached.layout.shapes[0].y0 == pytest.approx(0.5)
    assert any("nicht erreicht" in a.text for a in missed.layout.annotations)
    assert not any("nicht erreicht" in a.text for a in reached.layout.annotations)


def test_trace_figure_downsamples_long_series():
    fig = Z.build_trace(list(range(10000)), 100, 50, True)
    assert len(fig.data[0].x) <= 2000


def test_ring_and_grid_pictures_colour_by_state():
    ring = Z.build_ring([1, 0, 0, 1, 1, 0])
    assert locked(ring) and list(ring.data[0].marker.color) == [C.COLOR_A, C.COLOR_B, C.COLOR_B, C.COLOR_A, C.COLOR_A, C.COLOR_B]
    grid = Z.build_grid([1, 0, 0, 1, 1, 0, 0, 0, 1], 3)
    assert locked(grid) and grid.data[0].z.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]] and grid.layout.yaxis.autorange == "reversed"


def test_count_figure():
    fig = Z.build_count(3, 10)
    assert locked(fig) and list(fig.data[0].y) == [3, 7]


def test_time_over_n_has_complete_line_and_ring_points_with_open_markers_for_capped_cells():
    rows = [{"n": 10, "complete_log10": 2.7, "rings": {1: fake_cell(5.0), 2: fake_cell(20.0, capped=3), 3: None, 4: None}},
            {"n": 20, "complete_log10": 4.95, "rings": {1: fake_cell(5.0), 2: fake_cell(16.0), 3: None, 4: None}}]
    fig = Z.build_time_over_n(rows, 0.1)
    assert locked(fig) and fig.data[0].name.startswith("Vollgraph") and list(fig.data[0].y) == [2.7, 4.95]
    ring2 = next(t for t in fig.data if t.name.startswith("Ring k = 2"))
    assert list(ring2.marker.symbol) == ["circle-open", "circle"] and list(ring2.y) == pytest.approx([math.log10(20), math.log10(16)])
    assert not any(t.name.startswith("Ring k = 3") for t in fig.data)


def test_rates_figure_has_three_groups():
    rows = X.e2_rate_rows(0.4)
    fig = Z.build_rates(rows)
    assert locked(fig) and [t.name for t in fig.data][0].startswith("gemessen") and len(fig.data) == 3 and len(fig.data[0].x) == len(C.RATE_EPS)


def test_neighbourhood_figure_skips_nan_and_draws_the_no_advantage_line():
    rows = [{"k": 1, "d": 2, "share": 0.1, "thr": 1, "mean": 5.0, "se": 0.1, "capped": 0, "total": 160, "ratio": 100.0, "complete": 500.0},
            {"k": 9, "d": 18, "share": 0.95, "thr": 8, "mean": float("nan"), "se": float("nan"), "capped": 160, "total": 160, "ratio": float("nan"), "complete": 500.0}]
    fig = Z.build_neighbourhood(rows, 20, 0.1)
    assert locked(fig) and len(fig.data[0].x) == 1 and fig.layout.yaxis.type == "log" and fig.layout.shapes[0].y0 == 1


def test_staircase_one_line_per_k_and_skips_missing_cells():
    rows = [{"k": 3, "q": 0.2, "thr": 2, "spread": 1, "mean": 3.0, "se": 0.1, "capped": 0, "total": 240},
            {"k": 3, "q": 0.3, "thr": 2, "spread": 1, "mean": None, "se": None, "capped": None, "total": None},
            {"k": 4, "q": 0.2, "thr": 2, "spread": 2, "mean": 5.0, "se": 0.1, "capped": 0, "total": 240}]
    fig = Z.build_staircase(rows)
    assert locked(fig) and [t.name for t in fig.data] == ["k = 3", "k = 4"] and len(fig.data[0].x) == 1


def test_grid_times_figure_lines_per_cell_and_filters_eps():
    rows = [{"kind": "vn", "q": 0.25, "thr": 1, "edge": 1, "eps": 0.2, "side": 4, "mean": 2.0, "se": 0.1, "capped": 0, "total": 240},
            {"kind": "vn", "q": 0.25, "thr": 1, "edge": 1, "eps": 0.1, "side": 4, "mean": 3.0, "se": 0.1, "capped": 0, "total": 240},
            {"kind": "moore", "q": 0.4, "thr": 4, "edge": 3, "eps": 0.2, "side": 4, "mean": 80.0, "se": 3.0, "capped": 4, "total": 240}]
    fig = Z.build_grid_times(rows, 0.2)
    by_name = {t.name: t for t in fig.data}
    assert locked(fig) and len(fig.data) == 2
    assert by_name["8 Nachbarn, q = 0,4 (Schwelle 4)"].marker.symbol == ("circle-open",)
    assert list(by_name["4 Nachbarn, q = 0,25 (Schwelle 1)"].y) == pytest.approx([math.log10(2.0)])      # ε = 0,1 ist herausgefiltert


def test_back_figure_two_lines_for_one_eps():
    rows = X.e6_rows(0.4)
    fig = Z.build_back(rows, 0.1)
    assert locked(fig) and len(fig.data) == 2 and len(fig.data[0].x) == len(C.BACK_NS) and all(b > h for h, b in zip(fig.data[0].y, fig.data[1].y))


def test_linear_vs_log_figure_colours_negative_results():
    rows = [{"n": 40, "log_time": 4e9, "linear": 4e9, "deviation": 1e-7, "negative": False}, {"n": 70, "log_time": 5e16, "linear": -7e15, "deviation": 1.1, "negative": True},
            {"n": 10, "log_time": 495.0, "linear": 495.0, "deviation": 0.0, "negative": False}]
    fig = Z.build_linear_vs_log(rows)
    assert locked(fig) and list(fig.data[0].marker.color) == [C.COLOR_LINE, C.COLOR_RED, C.COLOR_LINE] and min(fig.data[0].y) >= 1e-16 and fig.layout.yaxis.type == "log"
