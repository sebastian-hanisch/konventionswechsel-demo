"""Das Messraster der vorgerechneten Experimente (E1, E3, E4, E5) und die Auswertung einer Zelle.

Nur Raster und Aggregation, keine Texte: der Erzeuger-Hash der Messdatei (data/measurements.json) deckt diese Datei und die Simulation ab; Textänderungen an der App machen die Daten nicht ungültig.
Jede Zelle läuft über vier feste Zufalls-Ströme (`SEEDS`); gemittelt wird über alle beendeten Läufe, die Streuung der vier Seed-Mittel ist mitgespeichert."""

from __future__ import annotations

import math

import numpy as np

import kvw_graphs as GR
import kvw_model as M
import kvw_sim as SI

SEEDS = (100001, 100002, 100003, 100004)      # feste Zufallsströme der Messreihen: nie ein einzelner Seed als Befund
Q_MEASURE = 0.4                                # Schwelle der Messreihen E1, E3, E5 (E4 variiert q)

E1_EPS = (0.2, 0.1, 0.05)
E1_KS = (1, 2, 3, 4)
E1_NS = (10, 20, 40, 80, 160, 320)
E1_RUNS, E1_CAP = 60, 4_000_000

E3_CONFIGS = ((20, 0.2, 9), (30, 0.2, 14), (40, 0.2, 8), (20, 0.1, 9))     # (n, ε, größtes k)
E3_RUNS, E3_CAP = 40, 6_000_000

E4_N, E4_EPS = 30, 0.2
E4_KS = (3, 4, 5, 6)
E4_QS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
E4_RUNS, E4_CAP = 60, 4_000_000

E5_SIDES = (4, 8, 12, 24)
E5_EPS = (0.2, 0.1)
E5_CELLS = (("vn", 0.25), ("vn", 0.40), ("moore", 0.30), ("moore", 0.40))   # (Nachbarschaft, q)
E5_RUNS, E5_CAP = 60, 4_000_000


def cell_specs() -> list[dict]:
    """Alle Zellen des Rasters, ohne Doppelungen: gleiche (Art, n, k, thr, ε) werden nur einmal gerechnet (gleiche Ströme, also zahlengleich)."""
    specs: dict[tuple, dict] = {}

    def add(experiment, kind, n, k, q, eps, runs, cap):
        thr = M.threshold(q, GR.degree(kind, n, k))
        key = (kind, n, k, thr, eps)
        if key not in specs:
            specs[key] = {"kind": kind, "n": n, "k": k, "thr": thr, "eps": eps, "runs": runs, "cap": cap, "experiments": []}
        if experiment not in specs[key]["experiments"]:
            specs[key]["experiments"].append(experiment)

    for eps in E1_EPS:
        for k in E1_KS:
            for n in E1_NS:
                add("e1", GR.KIND_RING, n, k, Q_MEASURE, eps, E1_RUNS, E1_CAP)
    for n, eps, kmax in E3_CONFIGS:
        for k in range(1, kmax + 1):
            add("e3", GR.KIND_RING, n, k, Q_MEASURE, eps, E3_RUNS, E3_CAP)
    for k in E4_KS:
        for q in E4_QS:
            add("e4", GR.KIND_RING, E4_N, k, q, E4_EPS, E4_RUNS, E4_CAP)
    for side in E5_SIDES:
        for eps in E5_EPS:
            for kind, q in E5_CELLS:
                add("e5", kind, side * side, None, q, eps, E5_RUNS, E5_CAP)
    return list(specs.values())


def measure_cell(spec: dict) -> dict:
    """Rechnet eine Zelle: je Seed `runs` Läufe, Schrittgrenze `cap`. Abgebrochene Läufe zählen nie als Zeit (`capped`); ist einer abgebrochen, ist das Mittel nach unten verzerrt."""
    nb = GR.network_nb(spec["kind"], spec["n"], spec["k"])
    per_seed, pooled, capped = [], [], 0
    for seed in SEEDS:
        res = SI.simulate(nb, spec["thr"], spec["eps"], spec["runs"], seed, spec["cap"])
        fin = res.finished
        capped += res.capped
        pooled.extend(float(v) for v in fin)
        per_seed.append(float(fin.mean()) if len(fin) else float("nan"))
    pooled_arr = np.array(pooled)
    mean, se = (float(pooled_arr.mean()), float(pooled_arr.std(ddof=1) / math.sqrt(len(pooled_arr)))) if len(pooled_arr) > 1 else (float("nan"), float("nan"))
    valid = [m for m in per_seed if m == m]
    return {
        **{key: spec[key] for key in ("kind", "n", "k", "thr", "eps", "runs", "cap", "experiments")},
        "d": GR.degree(spec["kind"], spec["n"], spec["k"]),
        "mean": mean, "se": se, "sd_seeds": float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan"),
        "median": float(np.median(pooled_arr)) if len(pooled_arr) else float("nan"),
        "p10": float(np.percentile(pooled_arr, 10)) if len(pooled_arr) else float("nan"),
        "p90": float(np.percentile(pooled_arr, 90)) if len(pooled_arr) else float("nan"),
        "finished": len(pooled), "capped": capped, "total": len(SEEDS) * spec["runs"],
    }
