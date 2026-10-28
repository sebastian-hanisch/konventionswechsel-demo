"""Szenario: aus den Reglern wird ein konkretes Netz mit Schwelle. Reine Funktionen, kein Zufall."""

from __future__ import annotations

import math
from dataclasses import dataclass

import kvw_constants as C
import kvw_graphs as GR
import kvw_model as M


def grid_side(n: int) -> int:
    """Kantenlänge L des Gitters zu n Betrieben: n wird auf die nächste Quadratzahl gerundet (ab 0,5 aufwärts), L zwischen GRID_SIDE_MIN und GRID_SIDE_MAX."""
    return max(C.GRID_SIDE_MIN, min(C.GRID_SIDE_MAX, int(math.sqrt(n) + 0.5)))


def effective_n(kind: str, n: int) -> int:
    """Zahl der Betriebe, mit der wirklich gerechnet wird: beim Gitter L², sonst n."""
    if kind in (GR.KIND_VN, GR.KIND_MOORE):
        return grid_side(n) ** 2
    return n


def effective_k(kind: str, n: int, k: int) -> int | None:
    """Nachbarn je Seite im Ring, begrenzt auf (n − 1)//2 (sonst wären Partner doppelt); für andere Netze None."""
    if kind != GR.KIND_RING:
        return None
    return max(C.K_MIN, min(k, (n - 1) // 2))


@dataclass(frozen=True)
class NetworkSpec:
    kind: str
    n: int                 # tatsächliche Zahl der Betriebe
    k: int | None          # Ring: Partner je Seite
    d: int                 # Partner je Betrieb
    thr: int               # nötige A-Partner
    edge: int | None       # A-Partner am geraden Rand eines A-Blocks (None im Vollgraph)

    @property
    def spread(self) -> int | None:
        """Spielraum s = Randsicht − thr: so viele A-Partner dürfen an einer geraden Kante fehlen, ohne dass sie stehen bleibt. Negativ: die Kante rückt ohne Fehler nicht vor."""
        return None if self.edge is None else self.edge - self.thr

    @property
    def share(self) -> float:
        """Anteil der Partner, die A führen müssen: thr/d."""
        return self.thr / self.d


def resolve_network(network: str, n: int, k: int, q: float) -> NetworkSpec:
    """Das Netz zu den Reglern: Art, tatsächliches n und k, Partnerzahl, ganzzahlige Schwelle und Randsicht."""
    kind = C.NETWORK_KIND[network]
    n_eff = effective_n(kind, n)
    k_eff = effective_k(kind, n_eff, k)
    d = GR.degree(kind, n_eff, k_eff)
    return NetworkSpec(kind, n_eff, k_eff, d, M.threshold(q, d), GR.edge_view(kind, k_eff))
