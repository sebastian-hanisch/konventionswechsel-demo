"""Exakte Wechselzeiten (Erwartungswerte), ohne Simulation.

Zeit wird in Aktualisierungen JE BETRIEB gemessen (Einzelaktualisierungen geteilt durch n). Drei Rechenwege:
  * Vollgraph, Geburts-Sterbe-Kette über die Zahl k der A-Betriebe, **im Logarithmus** (`complete_log_time`): trägt auch Zeiten wie 10^240, die sich nicht ausschreiben lassen.
  * Vollgraph als lineares System (`complete_linear_time`): nur ein Lehrstück und Orakel; für große n läuft es in Rundungsfehler (E7) und verweigert sich ab n > 40.
  * Beliebiges kleines Netz über alle 2^n Belegungen (`graph_exact_time`, dichtes numpy): live bis n = 12."""

from __future__ import annotations

import math

import numpy as np

import kvw_model as M

LINEAR_SAFE_MAX_N = 40         # darüber verweigert `complete_linear_time` ohne `unsafe=True` (E7 zeigt, warum)
EXACT_MAX_N = 12               # 2^12 = 4 096 Belegungen, höchstens 1 586 Übergangszustände: dichtes numpy, rund eine Sekunde
LN10 = math.log(10.0)


def kl(q: float, p: float) -> float:
    """Relative Entropie KL(q ‖ p) = q·ln(q/p) + (1 − q)·ln((1 − q)/(1 − p)) in Nat."""
    if not (0.0 < q < 1.0 and 0.0 < p < 1.0):
        raise ValueError(f"q = {q}, p = {p}: beide müssen in (0, 1) liegen")
    return q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))


def chernoff_rate(q: float, eps: float) -> float:
    """Wachstumsrate von log10 der Wechselzeit je Betrieb im Vollgraph: KL(q ‖ ε/2)/ln 10. Herleitung: unterhalb der Schwelle steigt die Kette nur durch Fehler (ε/2 je Betrieb),
    die stationären Gewichte sind dort binomial Bin(n, ε/2); die Wartezeit ist ungefähr 1/P(Bin(n, ε/2) ≥ thr), und ln P ≈ −n·KL(thr/n ‖ ε/2) (Chernoff)."""
    return kl(q, eps / 2.0) / LN10


def rough_rate(q: float, eps: float) -> float:
    """Die grobe Erwartung q·log10(1/ε) (jeder der q·n nötigen Betriebe müsste einen Fehler machen): sie überschätzt die Rate, weil ein Fehler mit Wahrscheinlichkeit ε/2 kommt UND
    die Betriebe nicht gezielt, sondern zufällig ausgewählt werden; gezeigt in E2."""
    return q * math.log10(1.0 / eps)


def birth_death_rates(n: int, thr: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Aufwärts- und Abwärtsrate je Einzelaktualisierung für k = 0 .. n A-Betriebe im Vollgraph.
    up[k]   = (n − k)/n · ((1 − ε)·[k ≥ thr] + ε/2)        ein B-Betrieb sieht k A-Partner;
    down[k] = k/n · ((1 − ε)·[k − 1 < thr] + ε/2)          ein A-Betrieb sieht k − 1 A-Partner."""
    if n < 2:
        raise ValueError(f"n = {n}: mindestens 2 Betriebe")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"ε = {eps} liegt außerhalb von 0 bis 1")
    k = np.arange(n + 1, dtype=np.float64)
    up = (n - k) / n * ((1.0 - eps) * (k >= thr) + eps / 2.0)
    down = k / n * ((1.0 - eps) * ((k - 1.0) < thr) + eps / 2.0)
    return up, down


def complete_log_time(n: int, thr: int, eps: float, target: int | None = None) -> float:
    """ln der erwarteten Aktualisierungen je Betrieb, bis im Vollgraph von „alle B“ aus erstmals mindestens `target` Betriebe A führen (Standard ⌈n/2⌉).
    E[Schritte] = Σ_{k<target} 1/(π_k·up_k) · Σ_{j≤k} π_j mit π_j ∝ Π up/down; alles im Logarithmus aufsummiert (logaddexp), damit nichts überläuft."""
    m = M.goal(n) if target is None else int(target)
    if not 1 <= m <= n:
        raise ValueError(f"target = {m} liegt außerhalb von 1 bis {n}")
    up, down = birth_death_rates(n, thr, eps)
    log_pi = np.concatenate([[0.0], np.cumsum(np.log(up[: m - 1]) - np.log(down[1:m]))])
    cum = np.logaddexp.accumulate(log_pi)
    terms = cum - log_pi - np.log(up[:m])
    return float(np.logaddexp.reduce(terms)) - math.log(n)


def complete_log10_time(n: int, thr: int, eps: float, target: int | None = None) -> float:
    """log10 der erwarteten Aktualisierungen je Betrieb (siehe `complete_log_time`)."""
    return complete_log_time(n, thr, eps, target) / LN10


def complete_linear_time(n: int, thr: int, eps: float, target: int | None = None, unsafe: bool = False) -> float:
    """Dieselbe erwartete Zeit als lineares System (I − P)·t = 1 über die Zustände k < target (dichtes numpy). Orakel für kleine n und Lehrstück: ab n > 40 verweigert die Funktion
    ihre Antwort (ValueError), weil Rundungsfehler die Lösung entwerten; E7 ruft sie mit `unsafe=True` auf, um genau das zu zeigen."""
    if n > LINEAR_SAFE_MAX_N and not unsafe:
        raise ValueError(f"n = {n} > {LINEAR_SAFE_MAX_N}: das lineare System ist dort numerisch unbrauchbar (Experiment E7); complete_log_time verwenden")
    m = M.goal(n) if target is None else int(target)
    if not 1 <= m <= n:
        raise ValueError(f"target = {m} liegt außerhalb von 1 bis {n}")
    up, down = birth_death_rates(n, thr, eps)
    a = np.zeros((m, m))
    for k in range(m):
        a[k, k] = up[k] + down[k]
        if k + 1 < m:
            a[k, k + 1] = -up[k]
        if k > 0:
            a[k, k - 1] = -down[k]
    t = np.linalg.solve(a, np.ones(m))
    return float(t[0] / n)


def mirror(n: int, thr: int) -> tuple[int, int]:
    """Rückwechsel im Vollgraph: Schwelle und Ziel der gespiegelten Kette (Rollen von A und B vertauscht, Start „alle A“). Gibt (thr_back, target_back)."""
    thr_back = M.mirror_threshold(n - 1, thr)
    return thr_back, M.mirror_target(n, thr_back)


def mirror_log_time(n: int, thr: int, eps: float) -> float:
    """ln der erwarteten Aktualisierungen je Betrieb für den Rückwechsel (alle A → thr_back + 2 B) im Vollgraph."""
    thr_back, target_back = mirror(n, thr)
    return complete_log_time(n, thr_back, eps, target_back)


def _popcount(states: np.ndarray, n: int) -> np.ndarray:
    pop = np.zeros(len(states), dtype=np.int64)
    for b in range(n):
        pop += (states >> b) & 1
    return pop


def graph_exact_time(nb: np.ndarray, thr: int, eps: float) -> float:
    """Erwartete Aktualisierungen je Betrieb von „alle B“ bis mindestens ⌈n/2⌉ A auf einem beliebigen Netz `nb`, über alle 2^n Belegungen (nur n <= EXACT_MAX_N).
    Übergangszustände sind die Belegungen mit weniger als ⌈n/2⌉ A; (I − P)·t = 1, P aus: gezogener Betrieb i (Wahrscheinlichkeit 1/n) wählt A mit (1 − ε)·[A-Partner ≥ thr] + ε/2."""
    n, _ = nb.shape
    if n > EXACT_MAX_N:
        raise ValueError(f"n = {n} > {EXACT_MAX_N}: der Zustandsraum 2^n ist live zu groß")
    m = M.goal(n)
    states = np.arange(1 << n, dtype=np.int64)
    trans = states[_popcount(states, n) < m]
    index = -np.ones(1 << n, dtype=np.int64)
    index[trans] = np.arange(len(trans))
    p = np.zeros((len(trans), len(trans)))
    rows = np.arange(len(trans))
    for i in range(n):
        count_a = np.zeros(len(trans), dtype=np.int64)
        for j in nb[i]:
            count_a += (trans >> int(j)) & 1
        p_a = (1.0 - eps) * (count_a >= thr) + eps / 2.0
        for target_state, prob in ((trans | (1 << i), p_a), (trans & ~(1 << i), 1.0 - p_a)):
            col = index[target_state]
            inside = col >= 0
            p[rows[inside], col[inside]] += prob[inside] / n
    t = np.linalg.solve(np.eye(len(trans)) - p, np.ones(len(trans)))
    return float(t[index[0]] / n)
