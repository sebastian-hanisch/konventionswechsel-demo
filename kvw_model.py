"""Die Entscheidungsregel eines Betriebs: beste Antwort mit Fehlerrate ε.

Ein Betrieb mit d Partnern führt das neue Format A, wenn mindestens der Anteil q seiner Partner es führt (Hirschjagd-Schwelle, q < 1/2: A ist risikodominant),
sonst das alte Format B. Mit Wahrscheinlichkeit ε wählt er stattdessen zufällig (A oder B je zur Hälfte). Die Zahl der nötigen A-Partner ist die ganzzahlige Schwelle thr."""

from __future__ import annotations

import math

TOLERANCE = 1e-9


def threshold(q: float, d: int) -> int:
    """thr = ⌈q·d⌉: so viele A-Partner braucht ein Betrieb mindestens. Die Toleranz 1e-9 verhindert, dass ein Gleitkomma-Rest (0,3·10 = 3,0000000000000004) die Schwelle um 1 anhebt;
    bei Gleichstand (q·d ganzzahlig) gewinnt A."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q = {q} liegt außerhalb von 0 bis 1")
    if d < 1:
        raise ValueError(f"d = {d}: ein Betrieb braucht mindestens einen Partner")
    return int(math.ceil(q * d - TOLERANCE))


def goal(n: int) -> int:
    """Ziel des Hinwechsels: mindestens die Hälfte der Betriebe führt A, ⌈n/2⌉."""
    if n < 2:
        raise ValueError(f"n = {n}: mindestens 2 Betriebe")
    return (n + 1) // 2


def best_reply_is_a(count_a: int, thr: int) -> bool:
    """Beste Antwort ohne Fehler: A genau dann, wenn mindestens thr Partner A führen."""
    return count_a >= thr


def prob_a(count_a: int, thr: int, eps: float) -> float:
    """Wahrscheinlichkeit, dass ein neu entscheidender Betrieb A wählt: (1 − ε) bei bester Antwort A, dazu ε/2 aus der Zufallswahl."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"ε = {eps} liegt außerhalb von 0 bis 1")
    return (1.0 - eps) * (1.0 if best_reply_is_a(count_a, thr) else 0.0) + eps / 2.0


def mirror_threshold(d: int, thr: int) -> int:
    """Rückwechsel (Rollen von A und B vertauscht): B ist beste Antwort genau dann, wenn weniger als thr Partner A führen, also mindestens d − thr + 1 Partner B führen."""
    return d - thr + 1


def mirror_target(n: int, thr_back: int) -> int:
    """Ziel des Rückwechsels im Vollgraph: thr_back + 2 B-Betriebe (höchstens n). Die Hälfte reicht nicht: unterhalb von thr_back fließt das System zurück zu „alle A“."""
    return min(n, thr_back + 2)
