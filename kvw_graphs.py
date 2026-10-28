"""Netze: Wer sieht wen? Nachbarlisten als Array (N Zeilen, d Spalten); jeder Betrieb hat gleich viele Partner (d-regulär).

Vollgraph = „offene Börse“ (jeder trifft jeden), Ring = „Korridor“ (k Partner je Seite), Gitter = „Regionen“ (Torus mit 4 oder 8 Nachbarn)."""

from __future__ import annotations

import numpy as np

KIND_COMPLETE, KIND_RING, KIND_VN, KIND_MOORE = "complete", "ring", "vn", "moore"
GRID_MIN_SIDE = 3          # kleinere Tori haben doppelte Nachbarn (L = 2: links = rechts)


def complete_nb(n: int) -> np.ndarray:
    """Vollgraph: Betrieb i hat alle anderen n - 1 Betriebe als Partner (aufsteigend sortiert)."""
    if n < 2:
        raise ValueError(f"n = {n}: ein Vollgraph braucht mindestens 2 Betriebe")
    all_ids = np.arange(n, dtype=np.int64)
    return np.array([np.delete(all_ids, i) for i in range(n)], dtype=np.int64)


def ring_nb(n: int, k: int) -> np.ndarray:
    """Ring: Betrieb i hat die k Nachbarn links und die k Nachbarn rechts (d = 2k). Braucht n >= 2k + 1, sonst wären Partner doppelt."""
    if k < 1:
        raise ValueError(f"k = {k}: mindestens ein Nachbar je Seite")
    if n < 2 * k + 1:
        raise ValueError(f"n = {n} ist zu klein für k = {k}: der Ring braucht n >= 2k + 1")
    offsets = [o for o in range(-k, k + 1) if o != 0]
    return np.array([[(i + o) % n for o in offsets] for i in range(n)], dtype=np.int64)


def torus_nb(lx: int, ly: int, kind: str) -> np.ndarray:
    """Torus lx × ly (Betrieb x·ly + y); kind 'vn' = 4 Nachbarn (oben, unten, links, rechts), 'moore' = 8 Nachbarn (mit Diagonalen)."""
    if lx < GRID_MIN_SIDE or ly < GRID_MIN_SIDE:
        raise ValueError(f"Kantenlängen {lx} × {ly}: ein Torus braucht mindestens {GRID_MIN_SIDE} je Richtung")
    if kind == KIND_VN:
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif kind == KIND_MOORE:
        offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    else:
        raise ValueError(f"unbekannte Nachbarschaft: {kind}")
    nb = np.zeros((lx * ly, len(offsets)), dtype=np.int64)
    for x in range(lx):
        for y in range(ly):
            for j, (a, b) in enumerate(offsets):
                nb[x * ly + y, j] = ((x + a) % lx) * ly + (y + b) % ly
    return nb


def edge_view(kind: str, k: int | None = None) -> int | None:
    """Wie viele Partner sieht ein Betrieb am geraden Rand eines A-Blocks als A? Ring: k (eine Seite), Gitter mit 4 Nachbarn: 1, mit 8 Nachbarn: 3.
    Im Vollgraph gibt es keinen Rand: None."""
    if kind == KIND_RING:
        if k is None or k < 1:
            raise ValueError("der Ring braucht k >= 1")
        return int(k)
    if kind == KIND_VN:
        return 1
    if kind == KIND_MOORE:
        return 3
    if kind == KIND_COMPLETE:
        return None
    raise ValueError(f"unbekannte Art: {kind}")


def side_of(kind: str, n: int) -> int:
    """Kantenlänge L des quadratischen Torus mit n = L² Betrieben."""
    side = int(round(n ** 0.5))
    if side * side != n or side < GRID_MIN_SIDE:
        raise ValueError(f"n = {n} ist keine Quadratzahl ab {GRID_MIN_SIDE}² für ein Gitter")
    return side


def network_nb(kind: str, n: int, k: int | None = None) -> np.ndarray:
    """Nachbarliste zur Art des Netzes: 'complete', 'ring' (mit k), 'vn' / 'moore' (n = L²)."""
    if kind == KIND_COMPLETE:
        return complete_nb(n)
    if kind == KIND_RING:
        if k is None:
            raise ValueError("der Ring braucht k")
        return ring_nb(n, k)
    if kind in (KIND_VN, KIND_MOORE):
        side = side_of(kind, n)
        return torus_nb(side, side, kind)
    raise ValueError(f"unbekannte Art: {kind}")


def degree(kind: str, n: int, k: int | None = None) -> int:
    """Zahl der Partner je Betrieb d, ohne die Liste zu bauen."""
    if kind == KIND_COMPLETE:
        return n - 1
    if kind == KIND_RING:
        if k is None:
            raise ValueError("der Ring braucht k")
        return 2 * k
    if kind == KIND_VN:
        return 4
    if kind == KIND_MOORE:
        return 8
    raise ValueError(f"unbekannte Art: {kind}")
