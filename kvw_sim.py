"""Simulation des Wechsels: viele Läufe gleichzeitig (numpy über die Läufe), aller Zufall aus SplitMix64 (kvw_rng.py).

Ein Schritt = ein gleichverteilt gezogener Betrieb entscheidet neu. Je Schritt werden drei Zahlen verbraucht (Betrieb, Fehler?, Münze), auch wenn die dritte nicht gebraucht wird:
  * u0 → Betrieb i = ⌊u0·n⌋;
  * u1 < ε → Zufallswahl, A genau dann, wenn u2 < ½;
  * sonst beste Antwort: A genau dann, wenn mindestens thr Partner A führen.
Jeder Lauf hat seinen eigenen SplitMix64-Strom (`stream_seed`); Läufe sind unabhängig und das Ergebnis hängt nicht an der Zahl der Läufe, die gleichzeitig mitlaufen.
Start immer „alle B“. Zeit = Schritte / n (Aktualisierungen je Betrieb). Läufe, die `cap_steps` Schritte erreichen, ohne am Ziel zu sein, zählen NICHT als Zeit (nan), sondern als abgebrochen."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kvw_model as M
import kvw_rng as R

CHUNK = 2048                 # Schritte je Zufallsblock
SEED_STRIDE = 1_000_003      # Abstand der Strom-Seeds verschiedener Läufe


def stream_seed(seed: int, run: int) -> int:
    """Seed des Stroms für Lauf `run` zum Zufalls-Seed `seed`."""
    return (seed * SEED_STRIDE + run) & R._MASK


def _block_seed(stream: int, chunk_index: int) -> int:
    """Seed, ab dem der Strom hinter `chunk_index` ganzen Blöcken weiterläuft: SplitMix64 ist ein Zähler (state += γ), also genügt eine Verschiebung um 3·CHUNK·γ je Block."""
    return (stream + 3 * CHUNK * chunk_index * R._GAMMA) & R._MASK


@dataclass(frozen=True)
class SimResult:
    times: np.ndarray        # Aktualisierungen je Betrieb je Lauf; nan = abgebrochen
    capped: int              # Zahl der abgebrochenen Läufe
    cap_steps: int           # Schrittgrenze je Lauf

    @property
    def finished(self) -> np.ndarray:
        return self.times[np.isfinite(self.times)]


def _draw_chunk(streams: list[int], chunk_index: int) -> np.ndarray:
    """Zufallszahlen eines Blocks für alle Läufe, Form (CHUNK, 3, Läufe)."""
    cols = [R.uniform_block(_block_seed(s, chunk_index), 3 * CHUNK).reshape(CHUNK, 3) for s in streams]
    return np.stack(cols, axis=2)


def simulate(nb: np.ndarray, thr: int, eps: float, runs: int, seed: int, cap_steps: int) -> SimResult:
    """Agentenweise Simulation auf dem Netz `nb` (Form (n, d)); alle Läufe gleichzeitig. Gibt die Wechselzeiten (Aktualisierungen je Betrieb) bis mindestens ⌈n/2⌉ A-Betriebe."""
    n, _ = nb.shape
    target = M.goal(n)
    streams = [stream_seed(seed, r) for r in range(runs)]
    x = np.zeros((runs, n), dtype=np.int8)
    cnt = np.zeros((runs, n), dtype=np.int16)        # A-Partner je Betrieb
    total = np.zeros(runs, dtype=np.int64)           # A-Betriebe je Lauf
    ids = np.arange(runs)
    times = np.full(runs, np.nan)
    step = 0
    chunk_index = 0
    while len(ids) and step < cap_steps:
        block = _draw_chunk(streams, chunk_index)
        chunk_index += 1
        for t in range(CHUNK):
            if step >= cap_steps or not len(ids):
                break
            step += 1
            u = block[t][:, ids]
            rows = np.arange(len(ids))
            who = (u[0] * n).astype(np.int64)
            best = cnt[rows, who] >= thr
            new = np.where(u[1] < eps, u[2] < 0.5, best).astype(np.int8)
            delta = new - x[rows, who]
            x[rows, who] = new
            total[ids] += delta
            cnt[rows[:, None], nb[who]] += delta[:, None].astype(np.int16)
            done = total[ids] >= target
            if done.any():
                times[ids[done]] = step / n
                keep = ~done
                ids, x, cnt = ids[keep], x[keep], cnt[keep]
    return SimResult(times, int(np.isnan(times).sum()), cap_steps)


def simulate_complete(n: int, thr: int, eps: float, runs: int, seed: int, cap_steps: int, target: int | None = None) -> SimResult:
    """Vollgraph, nur die Zahl k der A-Betriebe wird verfolgt (gleiche Verteilung wie `simulate`, aber ohne Nachbarlisten: n bis 1 000 und mehr). Der gezogene Betrieb ist mit
    Wahrscheinlichkeit k/n ein A-Betrieb (⌊u0·n⌋ < k) und sieht dann k − 1 A-Partner, sonst k. `target` ersetzt das Ziel ⌈n/2⌉ (Rückwechsel: gespiegelte Schwelle und Ziel)."""
    goal = M.goal(n) if target is None else int(target)
    streams = [stream_seed(seed, r) for r in range(runs)]
    k = np.zeros(runs, dtype=np.int64)
    ids = np.arange(runs)
    times = np.full(runs, np.nan)
    step = 0
    chunk_index = 0
    while len(ids) and step < cap_steps:
        block = _draw_chunk(streams, chunk_index)
        chunk_index += 1
        for t in range(CHUNK):
            if step >= cap_steps or not len(ids):
                break
            step += 1
            u = block[t][:, ids]
            kk = k[ids]
            is_a = (u[0] * n) < kk
            count_a = np.where(is_a, kk - 1, kk)
            best = count_a >= thr
            new = np.where(u[1] < eps, u[2] < 0.5, best)
            k[ids] = kk + new.astype(np.int64) - is_a.astype(np.int64)
            done = k[ids] >= goal
            if done.any():
                times[ids[done]] = step / n
                ids = ids[~done]
    return SimResult(times, int(np.isnan(times).sum()), cap_steps)


def simulate_one(nb: np.ndarray, thr: int, eps: float, stream: int, cap_steps: int) -> tuple[float, list[int]]:
    """Ein Lauf als einfache Schleife (Nachvollziehbarkeit vor Tempo; zugleich das Orakel des vektorisierten `simulate`: dieselben Zahlen, dieselbe Zeit).
    Gibt (Zeit in Aktualisierungen je Betrieb, nan bei Abbruch; Zahl der A-Betriebe nach jedem Schritt)."""
    n, _ = nb.shape
    target = M.goal(n)
    neighbours = nb.tolist()
    state = [0] * n
    cnt = [0] * n
    k = 0
    series: list[int] = []
    chunk_index = 0
    step = 0
    while step < cap_steps:
        flat = R.uniform_block(_block_seed(stream, chunk_index), 3 * CHUNK).tolist()
        chunk_index += 1
        for t in range(CHUNK):
            if step >= cap_steps:
                break
            step += 1
            who = int(flat[3 * t] * n)
            if flat[3 * t + 1] < eps:
                new = 1 if flat[3 * t + 2] < 0.5 else 0
            else:
                new = 1 if cnt[who] >= thr else 0
            if new != state[who]:
                delta = new - state[who]
                state[who] = new
                k += delta
                for j in neighbours[who]:
                    cnt[j] += delta
            series.append(k)
            if k >= target:
                return step / n, series
    return float("nan"), series


def snapshots(nb: np.ndarray, thr: int, eps: float, stream: int, steps: list[int]) -> list[list[int]]:
    """Belegungen (0 = B, 1 = A) nach den angegebenen Schrittzahlen (aufsteigend) im selben Strom wie `simulate_one`: das Bild zum Verlauf."""
    n, _ = nb.shape
    wanted = sorted(set(int(s) for s in steps))
    wanted_set = set(wanted)
    neighbours = nb.tolist()
    state = [0] * n
    cnt = [0] * n
    out: dict[int, list[int]] = {}
    if not wanted:
        return []
    if wanted[0] == 0:
        out[0] = list(state)
    chunk_index = 0
    step = 0
    last = wanted[-1]
    while step < last:
        flat = R.uniform_block(_block_seed(stream, chunk_index), 3 * CHUNK).tolist()
        chunk_index += 1
        for t in range(CHUNK):
            if step >= last:
                break
            step += 1
            who = int(flat[3 * t] * n)
            if flat[3 * t + 1] < eps:
                new = 1 if flat[3 * t + 2] < 0.5 else 0
            else:
                new = 1 if cnt[who] >= thr else 0
            if new != state[who]:
                delta = new - state[who]
                state[who] = new
                for j in neighbours[who]:
                    cnt[j] += delta
            if step in wanted_set:
                out[step] = list(state)
    return [out[s] for s in wanted]


def mean_and_error(times: np.ndarray) -> tuple[float, float]:
    """Mittel und Standardfehler der beendeten Läufe (nan-Einträge werden NICHT mitgemittelt, die Zahl der abgebrochenen Läufe meldet der Aufrufer)."""
    finished = np.asarray(times, dtype=float)
    finished = finished[np.isfinite(finished)]
    if len(finished) == 0:
        return float("nan"), float("nan")
    if len(finished) == 1:
        return float(finished[0]), float("nan")
    return float(finished.mean()), float(finished.std(ddof=1) / math.sqrt(len(finished)))
