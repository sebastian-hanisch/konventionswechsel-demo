"""Rechnet die vorgerechneten Messreihen (E1, E3, E4, E5) neu und schreibt data/measurements.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_measurements.py [--jobs 6]
Ändert sich eine der Dateien kvw_grid.py, kvw_sim.py, kvw_graphs.py, kvw_model.py oder kvw_rng.py, stimmt der Hash nicht mehr (tests/test_measurements.py schlägt an,
die App zeigt dann keine vorgerechneten Zellen); danach diesen Aufruf wiederholen. Dauer: rund eine Viertelstunde bei 6 Prozessen (Läufe vektorisiert, Zellen auf Prozesse verteilt)."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvw_experiments as X  # noqa: E402
import kvw_grid as GRID  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6, help="Zahl der Prozesse")
    args = parser.parse_args()
    specs = GRID.cell_specs()
    # die langsamsten Zellen zuerst (große Zeit ~ Nachbarschaft), damit die Prozesse gleichmäßig auslasten
    specs.sort(key=lambda s: -(s["n"] * s["cap"] if s["experiments"] != ["e1"] else s["n"]))
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        cells = list(pool.map(GRID.measure_cell, specs, chunksize=1))
    cells.sort(key=lambda c: (c["kind"], c["n"], c["k"] or 0, c["thr"], c["eps"]))
    X.write_measurements(cells)
    print(f"geschrieben: {X.MEASUREMENTS_FILE}  ({len(cells)} Zellen, {time.time() - start:.0f} s, Hash {X.generator_hash()})")


if __name__ == "__main__":
    main()
