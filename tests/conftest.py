"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini3():
    """Vollgraph mit n = 3 Betrieben, q = 0,4: d = 2, thr = ⌈0,8⌉ = 1, Ziel ⌈3/2⌉ = 2.
    Zustände k = 0 und 1 (k = 2 ist das Ziel). Aufwärtsrate aus 0: (3/3)·ε/2 = ε/2; aus 1: (2/3)·(1 − ε/2); Abwärtsrate aus 1: (1/3)·(1 − ε/2).
    Schritte: t0 = 2/ε + t1, t1 = 1/(1 − ε/2) + t0/3, also t0 = 3/ε + 3/(2(1 − ε/2)). Zeit je Betrieb = t0/3 = 1/ε + 1/(2(1 − ε/2)):
    ε = 0,2: 5 + 1/1,8 = 5,5556;  ε = 0,1: 10 + 1/1,9 = 10,5263."""
    return {"n": 3, "q": 0.4, "thr": 1, "goal": 2, "time": {0.2: 5 + 1 / 1.8, 0.1: 10 + 1 / 1.9}}


@pytest.fixture
def ring4():
    """Ring mit n = 4, k = 1 (d = 2), q = 0,4: thr = 1, Ziel 2, ε = 0,2. Zustände: alle B, ein A, Ziel.
    Aus alle B: jeder gezogene Betrieb wird mit ε/2 = 0,1 zu A: t0 = 1 + 0,9·t0 + 0,1·t1, also t0 = 10 + t1.
    Aus ein A (Betrieb 0): die zwei Nachbarn (Wahrscheinlichkeit 1/2) sehen 1 A und wählen A mit 0,8 + 0,1 = 0,9; der Gegenüberliegende (1/4) sieht 0 und wählt A mit 0,1;
    das ist 1/2·0,9 + 1/4·0,1 = 0,475 zum Ziel. Zurück: Betrieb 0 (1/4) sieht 0 A und wird B mit 0,9: 0,225. Bleiben: 0,3.
    0,7·t1 = 1 + 0,225·t0 = 1 + 0,225·(10 + t1) → 0,475·t1 = 3,25, t1 = 6,8421, t0 = 16,8421 Schritte; je Betrieb /4 = 4,2105."""
    return {"n": 4, "k": 1, "q": 0.4, "thr": 1, "eps": 0.2, "time": 3.25 / 0.475 / 4 + 10 / 4}
