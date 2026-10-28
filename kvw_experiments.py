"""Experimente E1 bis E7: Tabellen und Reihen aus dem Kern (exakt, live) und aus den vorgerechneten Messreihen (data/measurements.json).

Vorgerechnet sind die Simulationen auf Ring und Gitter (E1, E3, E4, E5): viele Läufe über vier feste Zufalls-Ströme, mit Erzeuger-Hash (tools/make_measurements.py).
Live und exakt sind die Vollgraph-Zeiten (E1, E2), der Rückwechsel (E6) und das Lehrstück zum linearen System (E7)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import kvw_constants as C
import kvw_exact as E
import kvw_graphs as GR
import kvw_grid as GRID
import kvw_model as M

ROOT = Path(__file__).resolve().parent
MEASUREMENTS_FILE = ROOT / "data" / "measurements.json"
HASHED_FILES = ("kvw_grid.py", "kvw_sim.py", "kvw_graphs.py", "kvw_model.py", "kvw_rng.py")


def generator_hash() -> str:
    """Hash der Dateien, aus deren Code die vorgerechneten Zellen stammen (Zeilenenden normalisiert, damit CRLF/LF denselben Hash geben)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0" + text + b"\0")
    return h.hexdigest()[:16]


def write_measurements(cells: list[dict]) -> dict:
    payload = {"generator_hash": generator_hash(), "seeds": list(GRID.SEEDS), "q": GRID.Q_MEASURE, "cells": cells}
    MEASUREMENTS_FILE.parent.mkdir(exist_ok=True)
    MEASUREMENTS_FILE.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


def load_measurements() -> dict | None:
    """Die vorgerechneten Zellen, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört (dann zeigt die App sie nicht)."""
    if not MEASUREMENTS_FILE.exists():
        return None
    data = json.loads(MEASUREMENTS_FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


def find_cell(cells: list[dict], kind: str, n: int, k: int | None, thr: int, eps: float) -> dict | None:
    """Die Zelle zu (Art, n, k, thr, ε), oder None."""
    for c in cells:
        if c["kind"] == kind and c["n"] == n and c["k"] == k and c["thr"] == thr and abs(c["eps"] - eps) < 1e-12:
            return c
    return None


# --- Hilfen ----------------------------------------------------------------------------------------------------------------------------------
def complete_log10(n: int, q: float, eps: float) -> float:
    """log10 der erwarteten Aktualisierungen je Betrieb im Vollgraph (Hinwechsel) zu n, q, ε."""
    return E.complete_log10_time(n, M.threshold(q, n - 1), eps)


def back_log10(n: int, q: float, eps: float) -> float:
    """log10 der erwarteten Aktualisierungen je Betrieb für den Rückwechsel im Vollgraph."""
    return E.mirror_log_time(n, M.threshold(q, n - 1), eps) / E.LN10


def slope_per_firm(f, n_low: int, n_high: int) -> float:
    """Steigung von log10 T je Betrieb zwischen n_low und n_high: (f(n_high) − f(n_low))/(n_high − n_low)."""
    return (f(n_high) - f(n_low)) / (n_high - n_low)


# --- E1: Zeit über n ---------------------------------------------------------------------------------------------------------------------------
def e1_rows(cells: list[dict], eps: float, q: float = GRID.Q_MEASURE) -> list[dict]:
    """Je n: exakte Zeit im Vollgraph (live) und die vorgerechneten Ringzellen k = 1 .. 4 zu (q = 0,4, ε)."""
    rows = []
    for n in GRID.E1_NS:
        rings = {}
        for k in GRID.E1_KS:
            thr = M.threshold(q, 2 * k)
            rings[k] = find_cell(cells, GR.KIND_RING, n, k, thr, eps)
        rows.append({"n": n, "complete_log10": complete_log10(n, q, eps), "rings": rings})
    return rows


# --- E2: welche Rate? ---------------------------------------------------------------------------------------------------------------------------
def e2_rate_rows(q: float, eps_list=C.RATE_EPS, n_low: int = C.RATE_N_LOW, n_high: int = C.RATE_N_HIGH) -> list[dict]:
    """Je ε: gemessene Steigung (exakt) gegen KL(q ‖ ε/2)/ln 10 und gegen die grobe Erwartung q·log10(1/ε)."""
    rows = []
    for eps in eps_list:
        measured = slope_per_firm(lambda n, e=eps: complete_log10(n, q, e), n_low, n_high)
        chernoff, rough = E.chernoff_rate(q, eps), E.rough_rate(q, eps)
        rows.append({"eps": eps, "measured": measured, "chernoff": chernoff, "rough": rough, "rough_over_measured": rough / measured, "diff": measured - chernoff})
    return rows


def e2_exponent_rows(q: float, ns=C.EXPONENT_NS, eps_pair=C.EXPONENT_EPS) -> list[dict]:
    """Je n: Exponent a in T ∝ ε^(−a) (Steigung von ln T gegen ln(1/ε) zwischen den beiden ε) gegen die nötige Zahl thr = k* von A-Partnern."""
    hi, lo = eps_pair
    rows = []
    for n in ns:
        thr = M.threshold(q, n - 1)
        a = (E.complete_log_time(n, thr, lo) - E.complete_log_time(n, thr, hi)) / math.log(hi / lo)
        rows.append({"n": n, "thr": thr, "a": a, "a_over_thr": a / thr})
    return rows


# --- E3: Nachbarschaftsgröße -------------------------------------------------------------------------------------------------------------------
def e3_rows(cells: list[dict], n: int, eps: float, q: float = GRID.Q_MEASURE) -> list[dict]:
    """Je k: Ringzeit (vorgerechnet) und ihr Verhältnis zur exakten Vollgraph-Zeit (Vorteil des Netzes)."""
    complete = math.exp(E.complete_log_time(n, M.threshold(q, n - 1), eps))
    rows = []
    chosen = [c for c in cells if c["kind"] == GR.KIND_RING and c["n"] == n and abs(c["eps"] - eps) < 1e-12 and "e3" in c["experiments"]]
    for c in sorted(chosen, key=lambda cell: cell["k"]):
        rows.append({"k": c["k"], "d": c["d"], "share": c["d"] / (n - 1), "thr": c["thr"], "mean": c["mean"], "se": c["se"], "capped": c["capped"], "total": c["total"],
                     "ratio": complete / c["mean"] if c["mean"] == c["mean"] else float("nan"), "complete": complete})
    return rows


# --- E4: Treppe der Schwelle ------------------------------------------------------------------------------------------------------------------
def e4_rows(cells: list[dict]) -> list[dict]:
    """Je (k, q): Schwelle thr, Spielraum s = k − thr und vorgerechnete Zeit (Ring n = 30, ε = 0,2)."""
    rows = []
    for k in GRID.E4_KS:
        for q in GRID.E4_QS:
            thr = M.threshold(q, 2 * k)
            c = find_cell(cells, GR.KIND_RING, GRID.E4_N, k, thr, GRID.E4_EPS)
            rows.append({"k": k, "q": q, "thr": thr, "spread": k - thr, "mean": None if c is None else c["mean"], "se": None if c is None else c["se"],
                         "capped": None if c is None else c["capped"], "total": None if c is None else c["total"]})
    return rows


# --- E5: Gitter -------------------------------------------------------------------------------------------------------------------------------
def e5_rows(cells: list[dict]) -> list[dict]:
    """Je (Nachbarschaft, q, ε, L): vorgerechnete Zeit; die Randsicht steht dabei."""
    rows = []
    for eps in GRID.E5_EPS:
        for kind, q in GRID.E5_CELLS:
            thr = M.threshold(q, GR.degree(kind, 9))
            for side in GRID.E5_SIDES:
                c = find_cell(cells, kind, side * side, None, thr, eps)
                rows.append({"kind": kind, "q": q, "thr": thr, "edge": GR.edge_view(kind), "eps": eps, "side": side, "mean": None if c is None else c["mean"],
                             "se": None if c is None else c["se"], "capped": None if c is None else c["capped"], "total": None if c is None else c["total"]})
    return rows


# --- E6: Rückwechsel --------------------------------------------------------------------------------------------------------------------------
def e6_rows(q: float, ns=C.BACK_NS, eps_list=C.BACK_EPS) -> list[dict]:
    """Je ε und n: log10 der Hin- und der Rückwechselzeit (exakt), ihr Verhältnis; je n ab dem zweiten die Steigungen je Betrieb."""
    rows = []
    for eps in eps_list:
        prev = None
        for n in ns:
            hin, back = complete_log10(n, q, eps), back_log10(n, q, eps)
            row = {"eps": eps, "n": n, "hin": hin, "back": back, "ratio": back / hin, "slope_hin": None, "slope_back": None}
            if prev is not None:
                span = n - prev["n"]
                row["slope_hin"], row["slope_back"] = (hin - prev["hin"]) / span, (back - prev["back"]) / span
            prev = row
            rows.append(row)
    return rows


def e6_rate_pair(q: float, eps: float, n_low: int, n_high: int) -> dict:
    """Steigungen von Hin- und Rückwechsel zwischen n_low und n_high gegen die Chernoff-Raten KL(q ‖ ε/2) bzw. KL(1 − q ‖ ε/2) (jeweils durch ln 10)."""
    return {"slope_hin": slope_per_firm(lambda n: complete_log10(n, q, eps), n_low, n_high), "slope_back": slope_per_firm(lambda n: back_log10(n, q, eps), n_low, n_high),
            "rate_hin": E.chernoff_rate(q, eps), "rate_back": E.chernoff_rate(1.0 - q, eps)}


# --- E7: warum im Logarithmus ----------------------------------------------------------------------------------------------------------------
def e7_rows(ns=C.LINEAR_NS, q: float = C.LINEAR_Q, eps: float = C.LINEAR_EPS) -> list[dict]:
    """Je n: Zeit im Logarithmus gegen das lineare System; Abweichung relativ, und ob das lineare System negativ wird."""
    rows = []
    for n in ns:
        thr = M.threshold(q, n - 1)
        log_time = math.exp(E.complete_log_time(n, thr, eps))
        linear = E.complete_linear_time(n, thr, eps, unsafe=True)
        rows.append({"n": n, "log_time": log_time, "linear": linear, "deviation": abs(linear - log_time) / log_time, "negative": linear < 0})
    return rows
