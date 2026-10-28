"""Der Verlauf eines einzelnen Laufs für die Hauptansicht (Bild): Lauf 0 des Zufalls-Seeds, Schritt für Schritt in einer einfachen Schleife.

Derselbe Strom wie in `kvw_sim` (Lauf 0), also dieselbe Zeit wie der erste Lauf der Simulation; die Tests prüfen das. Im Vollgraph wird nur die Zahl der A-Betriebe verfolgt (gezählte Fassung)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import kvw_graphs as GR
import kvw_model as M
import kvw_rng as R
import kvw_sim as SI


@dataclass(frozen=True)
class Trace:
    time: float                 # Aktualisierungen je Betrieb bis zum Ziel; nan, wenn die Schrittgrenze zuerst erreicht wurde
    series: list[int]           # Zahl der Betriebe der neuen Richtung nach jedem Schritt
    target: int
    reached: bool


def trace_complete(n: int, thr: int, eps: float, stream: int, cap_steps: int, target: int | None = None) -> Trace:
    """Vollgraph, gezählte Fassung als einfache Schleife (dieselben Zahlen wie `simulate_complete` für diesen Strom)."""
    goal = M.goal(n) if target is None else int(target)
    k = 0
    series: list[int] = []
    chunk_index = 0
    step = 0
    while step < cap_steps:
        flat = R.uniform_block(SI._block_seed(stream, chunk_index), 3 * SI.CHUNK).tolist()
        chunk_index += 1
        for t in range(SI.CHUNK):
            if step >= cap_steps:
                break
            step += 1
            is_a = flat[3 * t] * n < k
            count_a = k - 1 if is_a else k
            new = (1 if flat[3 * t + 2] < 0.5 else 0) if flat[3 * t + 1] < eps else (1 if count_a >= thr else 0)
            k += new - (1 if is_a else 0)
            series.append(k)
            if k >= goal:
                return Trace(step / n, series, goal, True)
    return Trace(float("nan"), series, goal, False)


def trace_run(kind: str, n: int, k: int | None, thr: int, eps: float, seed: int, cap_steps: int, target: int | None = None) -> Trace:
    """Lauf 0 zum Zufalls-Seed: im Vollgraph gezählt (mit optionalem Ziel für den Rückwechsel), sonst agentenweise."""
    stream = SI.stream_seed(seed, 0)
    if kind == GR.KIND_COMPLETE:
        return trace_complete(n, thr, eps, stream, cap_steps, target)
    time, series = SI.simulate_one(GR.network_nb(kind, n, k), thr, eps, stream, cap_steps)
    return Trace(time, series, M.goal(n), time == time)


def frame_steps(total: int, count: int) -> list[int]:
    """Bis zu `count` aufsteigende Schrittzahlen von 0 bis `total` (beide enthalten), gleichmäßig verteilt."""
    if total < 0 or count < 2:
        raise ValueError("total >= 0 und count >= 2 nötig")
    steps = np.unique(np.linspace(0, total, count).round().astype(int))
    return [int(s) for s in steps]


def frames(kind: str, n: int, k: int | None, thr: int, eps: float, seed: int, steps: list[int]) -> list[list[int]]:
    """Belegungen (0 = B, 1 = A) nach den angegebenen Schritten im Strom von Lauf 0 (nur Ring und Gitter)."""
    if kind == GR.KIND_COMPLETE:
        raise ValueError("im Vollgraph gibt es kein Bild der Belegung, nur den Zähler")
    return SI.snapshots(GR.network_nb(kind, n, k), thr, eps, SI.stream_seed(seed, 0), steps)
