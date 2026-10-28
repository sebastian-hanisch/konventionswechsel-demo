"""Die vorgerechneten Messreihen (data/measurements.json) gehören zum aktuellen Rechencode.

Eigene Datei, weil jede Änderung an kvw_grid.py, kvw_sim.py, kvw_graphs.py, kvw_model.py oder kvw_rng.py den Erzeuger-Hash ändert: so bleibt die Meldung "Messdatei neu erzeugen" von den Tests der Rechnung getrennt.
"""

import json

import pytest

import kvw_experiments as X
import kvw_grid as GRID


def stored():
    return json.loads(X.MEASUREMENTS_FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash():
    assert stored()["generator_hash"] == X.generator_hash(), "kvw_grid/sim/graphs/model/rng wurde geändert: python tools/make_measurements.py ausführen"
    assert X.load_measurements() is not None


def test_load_returns_none_for_a_stale_hash(monkeypatch):
    monkeypatch.setattr(X, "generator_hash", lambda: "0000000000000000")
    assert X.load_measurements() is None


def test_load_returns_none_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(X, "MEASUREMENTS_FILE", tmp_path / "fehlt.json")
    assert X.load_measurements() is None


def test_the_file_has_exactly_the_cells_of_the_grid():
    data = stored()
    want = {(s["kind"], s["n"], s["k"], s["thr"], s["eps"]) for s in GRID.cell_specs()}
    have = {(c["kind"], c["n"], c["k"], c["thr"], c["eps"]) for c in data["cells"]}
    assert want == have and len(data["cells"]) == len(have) and data["seeds"] == list(GRID.SEEDS)


def test_run_counts_are_consistent_in_every_cell():
    for c in stored()["cells"]:
        assert c["total"] == len(GRID.SEEDS) * c["runs"] and c["finished"] + c["capped"] == c["total"], c
        if c["finished"] > 1:
            assert c["mean"] > 0 and c["se"] > 0 and c["p10"] <= c["median"] <= c["p90"]


def test_grid_has_no_duplicate_cells_and_every_experiment_is_covered():
    specs = GRID.cell_specs()
    keys = [(s["kind"], s["n"], s["k"], s["thr"], s["eps"]) for s in specs]
    assert len(keys) == len(set(keys))
    assert {e for s in specs for e in s["experiments"]} == {"e1", "e3", "e4", "e5"}


@pytest.mark.parametrize("kind,n,k,thr,eps", [("ring", 20, 1, 1, 0.2), ("ring", 10, 2, 2, 0.2), ("vn", 16, None, 1, 0.2)])
def test_a_few_cells_are_reproduced_exactly_by_the_current_code(kind, n, k, thr, eps):
    """Neuberechnung dreier billiger Zellen: gleicher Code, gleiche Ströme, gleiche Zahlen."""
    old = next(c for c in stored()["cells"] if (c["kind"], c["n"], c["k"], c["thr"], c["eps"]) == (kind, n, k, thr, eps))
    spec = next(s for s in GRID.cell_specs() if (s["kind"], s["n"], s["k"], s["thr"], s["eps"]) == (kind, n, k, thr, eps))
    new = GRID.measure_cell(spec)
    assert new["mean"] == pytest.approx(old["mean"], rel=1e-9) and new["capped"] == old["capped"] and new["finished"] == old["finished"]


def test_generator_hash_ignores_line_endings(tmp_path, monkeypatch):
    before = X.generator_hash()
    monkeypatch.setattr(X, "ROOT", tmp_path)
    for name in X.HASHED_FILES:
        text = (X.Path(__file__).resolve().parent.parent / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(text.replace(b"\n", b"\r\n"))
    assert X.generator_hash() == before
