"""Experimente: Reihen aus dem Kern (exakt) und aus den vorgerechneten Zellen; Zweig-Tests (jede Spalte weicht ab, keine Nullspalte)."""

import math

import pytest

import kvw_constants as C
import kvw_exact as E
import kvw_experiments as X
import kvw_grid as GRID
import kvw_model as M


@pytest.fixture(scope="module")
def cells():
    return X.load_measurements()["cells"]


def test_find_cell_matches_all_five_keys(cells):
    c = X.find_cell(cells, "ring", 20, 1, 1, 0.2)
    assert c is not None and c["n"] == 20 and c["thr"] == 1
    assert X.find_cell(cells, "ring", 20, 1, 2, 0.2) is None and X.find_cell(cells, "ring", 21, 1, 1, 0.2) is None
    assert X.find_cell(cells, "vn", 20, 1, 1, 0.2) is None and X.find_cell(cells, "ring", 20, 1, 1, 0.3) is None


def test_e1_rows_pair_the_exact_complete_time_with_the_ring_cells(cells):
    rows = X.e1_rows(cells, 0.1)
    assert [r["n"] for r in rows] == list(GRID.E1_NS)
    for r in rows:
        assert r["complete_log10"] == pytest.approx(E.complete_log10_time(r["n"], M.threshold(0.4, r["n"] - 1), 0.1))
        assert all(r["rings"][k] is not None and r["rings"][k]["thr"] == M.threshold(0.4, 2 * k) for k in GRID.E1_KS)
    assert [round(10 ** rows[i]["complete_log10"]) for i in (0, 1)] == [496, 89448]


def test_e1_rows_without_measurements_have_empty_rings():
    rows = X.e1_rows([], 0.1)
    assert all(v is None for r in rows for v in r["rings"].values()) and rows[0]["complete_log10"] > 2


def test_e1_every_ring_column_differs_from_the_next(cells):
    """Nullspalten-Regel: jede Ringspalte (k = 1 bis 4) weicht von der Nachbarspalte ab, und alle Zeiten sind positiv."""
    rows = X.e1_rows(cells, 0.1)
    for a, b in ((1, 2), (2, 3), (3, 4)):
        assert all(r["rings"][a]["mean"] < r["rings"][b]["mean"] for r in rows if r["n"] >= 20)
    assert all(r["rings"][k]["mean"] > 0 for r in rows for k in GRID.E1_KS)


@pytest.mark.parametrize("q", [0.1, 0.2, 0.3, 0.4, 0.45])
def test_e2_measured_slope_is_the_chernoff_rate_for_every_q(q):
    row = X.e2_rate_rows(q, (0.1,), 400, 800)[0]
    assert abs(row["diff"]) < 0.0005 and row["rough"] > 1.2 * row["measured"]


def test_e2_rows_for_the_default_q():
    rows = X.e2_rate_rows(0.4)
    assert [r["eps"] for r in rows] == list(C.RATE_EPS)
    assert all(abs(r["diff"]) < 0.0003 for r in rows) and [round(r["rough_over_measured"], 2) for r in rows] == [2.07, 1.66, 1.47, 1.33]
    assert all(r["measured"] > 0 for r in rows) and rows[0]["measured"] < rows[1]["measured"] < rows[2]["measured"] < rows[3]["measured"]


def test_e2_exponent_is_about_a_tenth_below_thr():
    rows = X.e2_exponent_rows(0.4)
    assert [r["thr"] for r in rows] == [4, 8, 16, 32] and all(0.91 < r["a_over_thr"] < 0.95 for r in rows)
    assert [round(r["a"], 2) for r in rows] == [3.67, 7.45, 15.01, 30.11]


def test_e3_rows_ratio_is_complete_over_ring(cells):
    rows = X.e3_rows(cells, 30, 0.2)
    assert [r["k"] for r in rows] == list(range(1, 15))
    for r in rows:
        assert r["ratio"] == pytest.approx(r["complete"] / r["mean"]) and r["d"] == 2 * r["k"] and r["share"] == pytest.approx(2 * r["k"] / 29)
    assert X.e3_rows(cells, 31, 0.2) == [] and X.e3_rows([], 30, 0.2) == []


def test_e3_marks_the_cell_with_aborted_runs(cells):
    rows = X.e3_rows(cells, 20, 0.1)
    assert rows[-1]["k"] == 9 and rows[-1]["capped"] > 0 and all(r["capped"] == 0 for r in rows[:-1])


def test_e4_cells_with_equal_k_and_thr_are_number_identical(cells):
    rows = X.e4_rows(cells)
    assert len(rows) == len(GRID.E4_KS) * len(GRID.E4_QS)
    for a in rows:
        for b in rows:
            if (a["k"], a["thr"]) == (b["k"], b["thr"]):
                assert a["mean"] == b["mean"]
    assert all(r["spread"] == r["k"] - r["thr"] and r["mean"] > 0 for r in rows)


def test_e4_time_rises_with_thr_at_fixed_k(cells):
    rows = X.e4_rows(cells)
    for k in GRID.E4_KS:
        by_thr = {}
        for r in rows:
            if r["k"] == k:
                by_thr[r["thr"]] = r["mean"]
        ordered = [by_thr[t] for t in sorted(by_thr)]
        assert ordered == sorted(ordered) and len(set(ordered)) == len(ordered)


def test_e5_rows_cover_all_cells_with_the_edge_view(cells):
    rows = X.e5_rows(cells)
    assert len(rows) == len(GRID.E5_EPS) * len(GRID.E5_CELLS) * len(GRID.E5_SIDES) and all(r["mean"] is not None for r in rows)
    assert {(r["kind"], r["thr"], r["edge"]) for r in rows} == {("vn", 1, 1), ("vn", 2, 1), ("moore", 3, 3), ("moore", 4, 3)}


def test_e6_rows_back_is_always_slower_and_slopes_start_at_the_second_n():
    rows = X.e6_rows(0.4)
    assert all(r["back"] > r["hin"] for r in rows) and len(rows) == len(C.BACK_NS) * len(C.BACK_EPS)
    assert sum(1 for r in rows if r["slope_hin"] is None) == len(C.BACK_EPS)
    assert all(r["slope_back"] > r["slope_hin"] for r in rows if r["slope_hin"] is not None)


def test_e6_rate_pair_matches_the_mirrored_chernoff_rate():
    pair = X.e6_rate_pair(0.4, 0.1, 320, 640)
    assert pair["slope_hin"] == pytest.approx(pair["rate_hin"], abs=0.001) and pair["slope_back"] == pytest.approx(pair["rate_back"], abs=0.001)


def test_e7_rows_deviation_grows_with_n_and_the_big_cases_break():
    rows = X.e7_rows()
    assert [r["n"] for r in rows] == list(C.LINEAR_NS)
    assert rows[0]["deviation"] < 1e-12 and next(r for r in rows if r["n"] == 30)["deviation"] < 1e-6
    assert next(r for r in rows if r["n"] == 60)["deviation"] > 1e-2 and next(r for r in rows if r["n"] == 70)["deviation"] > 0.5
    assert all(math.isfinite(r["linear"]) for r in rows)
