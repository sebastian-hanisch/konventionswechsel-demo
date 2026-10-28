"""Kennzahlen der Hauptansicht, Simulationsläufe der App und Verdikte (reine Funktionen: Art und Text, die App ruft `getattr(st, Art)(Text)`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import kvw_constants as C
import kvw_exact as E
import kvw_graphs as GR
import kvw_model as M
import kvw_scenario as S
import kvw_sim as SI

Z_LIMIT = 4.0                 # Marge der Prüfung Formel gegen Simulation: Standardfehler
LN10 = E.LN10


# --- Formate ----------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def fmt_log10(l10: float) -> str:
    """Eine Zeit aus ihrem Zehnerlogarithmus: unter 1 mit zwei Stellen, unter 10^5 als Zahl, bis 10^100 als Mantisse·10^Exponent, darüber als 10^x (die Zahl ließe sich nicht ausschreiben)."""
    if l10 < 0:
        return de(10 ** l10, 2)
    if l10 < 2:
        return de(10 ** l10, 1)
    if l10 < 5:
        return thousands(10 ** l10)
    if l10 < 100:
        exponent = int(math.floor(l10))
        mantissa = 10 ** (l10 - exponent)
        if round(mantissa, 1) >= 10:
            mantissa, exponent = 1.0, exponent + 1
        return f"{de(mantissa, 1)}·10^{exponent}"
    return f"10^{de(l10, 1)}"


def fmt_time(t: float) -> str:
    """Eine gewöhnliche Zeit (Aktualisierungen je Betrieb)."""
    if not (t == t) or t <= 0:
        return "–"
    return fmt_log10(math.log10(t))


def fmt_years(l10_updates: float) -> str:
    """Einordnung: so viele Jahre, wenn jeder Betrieb einmal pro Woche prüft (52 Wochen je Jahr)."""
    l10_years = l10_updates - math.log10(C.WEEKS_PER_YEAR)
    if l10_years < 0:
        return f"{de(10 ** l10_years * 12, 1)} Monate"
    return fmt_log10(l10_years) + " Jahre"


# --- Einstellungen und Hauptansicht -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Settings:
    network: str = C.NETWORKS[0]
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    q: float = C.DEFAULT_Q
    eps: float = C.DEFAULT_EPS
    direction: str = C.DIRECTIONS[0]
    runs: int = C.DEFAULT_RUNS
    budget: int = C.DEFAULT_BUDGET
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class MainView:
    settings: Settings
    spec: S.NetworkSpec
    back: bool                    # Rückwechsel gerechnet (nur im Vollgraph)
    direction_ignored: bool       # Rückwechsel gewählt, aber das Netz ist kein Vollgraph: gezeigt wird der Hinwechsel
    thr: int                      # Schwelle der gerechneten Kette
    target: int                   # Ziel: so viele Betriebe der neuen Richtung
    log10_time: float | None      # exakt im Vollgraph (Aktualisierungen je Betrieb), sonst None
    budget_time: float            # Schrittgrenze in Aktualisierungen je Betrieb

    @property
    def exceeds_budget(self) -> bool | None:
        """Liegt die exakte Zeit über der Schrittgrenze der Simulation? None ohne exakte Zeit."""
        return None if self.log10_time is None else self.log10_time > math.log10(self.budget_time)


def analyse_main(settings: Settings) -> MainView:
    """Netz, Schwelle, Ziel und (im Vollgraph) die exakte Wechselzeit zu den Reglern; leicht genug für jeden Rerun."""
    spec = S.resolve_network(settings.network, settings.n, settings.k, settings.q)
    wants_back = settings.direction == C.DIRECTIONS[1]
    back = wants_back and spec.kind == GR.KIND_COMPLETE
    ignored = wants_back and not back
    if spec.kind == GR.KIND_COMPLETE:
        if back:
            thr, target = E.mirror(spec.n, spec.thr)
            log10_time = E.complete_log_time(spec.n, thr, settings.eps, target) / LN10
        else:
            thr, target = spec.thr, M.goal(spec.n)
            log10_time = E.complete_log10_time(spec.n, thr, settings.eps)
    else:
        thr, target, log10_time = spec.thr, M.goal(spec.n), None
    return MainView(settings, spec, back, ignored, thr, target, log10_time, settings.budget / spec.n)


def exact_small_log10(spec: S.NetworkSpec, eps: float) -> float | None:
    """Exakte Zeit (log10) für ein Netz, das kein Vollgraph ist, über alle 2^n Belegungen; None, wenn n zu groß ist."""
    if spec.kind == GR.KIND_COMPLETE or spec.n > E.EXACT_MAX_N:
        return None
    return math.log10(E.graph_exact_time(GR.network_nb(spec.kind, spec.n, spec.k), spec.thr, eps))


# --- Simulation der Hauptansicht ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimSummary:
    runs: int
    finished: int
    capped: int
    mean: float
    se: float
    median: float
    p10: float
    p90: float
    cap_time: float               # Schrittgrenze in Aktualisierungen je Betrieb
    exact: float | None           # exakte Zeit (Aktualisierungen je Betrieb), wenn als Zahl darstellbar
    z: float | None               # (Mittel − exakt)/Standardfehler


def summarise(times: np.ndarray, cap_steps: int, n: int, exact: float | None = None) -> SimSummary:
    """Fasst die Zeiten der Läufe zusammen; abgebrochene Läufe (nan) zählen nur in `capped`, nie im Mittel."""
    runs = len(times)
    fin = np.asarray(times, dtype=float)
    fin = fin[np.isfinite(fin)]
    mean, se = SI.mean_and_error(times)
    z = None
    if exact is not None and se == se and se > 0:
        z = (mean - exact) / se
    if len(fin) == 0:
        median = p10 = p90 = float("nan")
    else:
        median, p10, p90 = float(np.median(fin)), float(np.percentile(fin, 10)), float(np.percentile(fin, 90))
    return SimSummary(runs, len(fin), runs - len(fin), mean, se, median, p10, p90, cap_steps / n, exact, z)


def run_main_simulation(view: MainView, exact_small: float | None = None) -> SimSummary:
    """Simulation zu den Reglern der Hauptansicht (Vollgraph: gezählte Fassung, sonst agentenweise); `exact_small` ist die exakte Zeit (Aktualisierungen je Betrieb) eines kleinen Netzes."""
    s, spec = view.settings, view.spec
    if spec.kind == GR.KIND_COMPLETE:
        res = SI.simulate_complete(spec.n, view.thr, s.eps, s.runs, s.seed, s.budget, view.target)
        exact = math.exp(view.log10_time * LN10) if view.log10_time is not None and view.log10_time < 300 else None
    else:
        res = SI.simulate(GR.network_nb(spec.kind, spec.n, spec.k), spec.thr, s.eps, s.runs, s.seed, s.budget)
        exact = exact_small
    return summarise(res.times, s.budget, spec.n, exact)


def check_cells() -> list[dict]:
    """Die Zellen der Prüfung Formel gegen Simulation: je ein Vollgraph, ein Ring und ein Gitter mit 12 Betrieben (klein genug für die exakte Rechnung über alle 2^12 Belegungen)."""
    return [
        {"label": "Vollgraph, n = 12", "nb": GR.complete_nb(12)},
        {"label": "Ring, n = 12, k = 2", "nb": GR.ring_nb(12, 2)},
        {"label": "Gitter 3 × 4, 8 Nachbarn", "nb": GR.torus_nb(3, 4, GR.KIND_MOORE)},
    ]


def run_check(q: float, eps: float, runs: int, seed: int, cap_steps: int) -> list[dict]:
    """Je Zelle: exakte Zeit (alle 2^n Belegungen) gegen die Simulation (agentenweise)."""
    rows = []
    for cell in check_cells():
        nb = cell["nb"]
        thr = M.threshold(q, nb.shape[1])
        exact = E.graph_exact_time(nb, thr, eps)
        res = SI.simulate(nb, thr, eps, runs, seed, cap_steps)
        sm = summarise(res.times, cap_steps, nb.shape[0], exact)
        rows.append({"label": cell["label"], "thr": thr, "exact": exact, "mean": sm.mean, "se": sm.se, "z": sm.z, "capped": sm.capped, "runs": runs})
    return rows


# --- Verdikte der Hauptansicht -----------------------------------------------------------------------------------------------------------------
def verdict_main(view: MainView) -> tuple[str, str]:
    """Kernfrage: Wie lange dauert der Wechsel? Im Vollgraph exakt, sonst der Hinweis auf die Simulation."""
    spec, s = view.spec, view.settings
    way = "Rückwechsel (alle A → B)" if view.back else "Hinwechsel (alle B → A)"
    if spec.kind != GR.KIND_COMPLETE:
        note = " Der Rückwechsel ist nur für den Vollgraph gerechnet; gezeigt ist der Hinwechsel." if view.direction_ignored else ""
        return "info", (f"Auf diesem Netz (n = {spec.n}, {spec.d} Partner, nötig sind {spec.thr}) gibt es keine geschlossene Formel; die Wechselzeit steht in der Simulation (Knopf unten) "
                        f"und, wenn n höchstens {E.EXACT_MAX_N} ist, exakt über alle 2^n Belegungen.{note}")
    first = (f"**{way} im Vollgraph (n = {spec.n}, jeder trifft jeden):** erwartet {fmt_log10(view.log10_time)} Aktualisierungen je Betrieb. "
             f"Dazu müssen {view.thr} von {spec.d} Partnern schon ihr Ziel führen, bevor es sich für den einzelnen lohnt; ohne Absprache entsteht das nur durch Fehler. "
             f"Zur Einordnung: bei einer Prüfung je Betrieb und Woche wären das {fmt_years(view.log10_time)}.")
    if view.exceeds_budget:
        return "warning", first + f" Das liegt über der Schrittgrenze der Simulation ({thousands(s.budget)} Schritte = {fmt_time(view.budget_time)} Aktualisierungen je Betrieb): Sie erreicht das Ziel nicht."
    return "success", first + f" Das liegt innerhalb der Schrittgrenze der Simulation ({thousands(s.budget)} Schritte = {fmt_time(view.budget_time)} Aktualisierungen je Betrieb)."


def verdict_spread(spec: S.NetworkSpec) -> tuple[str, str]:
    """Rand des A-Blocks: Kommt eine gerade Kante ohne Fehler voran? Der Spielraum s = Randsicht − Schwelle."""
    if spec.edge is None:
        return "info", (f"Im Vollgraph gibt es keinen Rand: Ein Betrieb braucht {spec.thr} von {spec.d} Partnern auf A ({de(100 * spec.share, 0)} %). "
                        "Solange weniger A führen, geht es zurück zu B; die Mehrheit kommt nur durch viele gleichzeitige Fehler zustande.")
    s = spec.spread
    if s >= 0:
        return "success", (f"Eine gerade Kante rückt ohne Fehler vor: Ein Betrieb am Rand eines A-Blocks sieht {spec.edge} A-Partner, nötig sind {spec.thr} (Spielraum s = {s}). "
                           "Ein A-Block, der groß genug ist, wächst dann von selbst weiter; wie groß der Keim sein muss, zeigt die Wechselzeit.")
    return "warning", (f"Eine gerade Kante rückt ohne Fehler nicht vor: Ein Betrieb am Rand sieht nur {spec.edge} A-Partner, nötig sind {spec.thr} (Spielraum s = {s}). "
                       "Jeder Schritt der Ausbreitung braucht einen zusätzlichen Fehler an der Kante.")


def verdict_simulation(sm: SimSummary) -> tuple[str, str]:
    """Ergebnis der Simulation: Mittel nur, wenn kein Lauf an der Schrittgrenze abgebrochen ist; sonst melden statt mitteln."""
    cap = f"{fmt_time(sm.cap_time)} Aktualisierungen je Betrieb"
    if sm.finished == 0:
        return "warning", (f"Kein Lauf von {sm.runs} erreichte das Ziel innerhalb der Schrittgrenze ({cap}). Die Wechselzeit liegt darüber; ein Mittel wird nicht gebildet."
                           + (f" Exakt erwartet: {fmt_time(sm.exact)}." if sm.exact is not None else ""))
    spread = f"Median {fmt_time(sm.median)}, 10 % der Läufe unter {fmt_time(sm.p10)}, 10 % über {fmt_time(sm.p90)}"
    if sm.capped:
        return "warning", (f"{sm.capped} von {sm.runs} Läufen erreichten das Ziel nicht innerhalb der Schrittgrenze ({cap}); das Mittel der {sm.finished} beendeten Läufe ({fmt_time(sm.mean)}) ist "
                           f"nach unten verzerrt und kein Schätzwert der Wechselzeit. {spread}.")
    text = f"**Simulation, {sm.runs} Läufe, alle am Ziel:** Mittel {fmt_time(sm.mean)} ± {fmt_time(sm.se)} (Standardfehler) Aktualisierungen je Betrieb. {spread}."
    if sm.exact is not None and sm.z is not None:
        if abs(sm.z) <= Z_LIMIT:
            return "success", text + f" Exakt erwartet: {fmt_time(sm.exact)}; Abweichung z = {de(sm.z, 1)} Standardfehler (Marge {de(Z_LIMIT, 0)})."
        return "warning", text + f" Exakt erwartet: {fmt_time(sm.exact)}; Abweichung z = {de(sm.z, 1)} Standardfehler, außerhalb der Marge von {de(Z_LIMIT, 0)}."
    return "info", text


def verdict_check(rows: list[dict]) -> tuple[str, str]:
    """Verdikt der Prüfung Formel gegen Simulation: wie viele Zellen innerhalb der Marge von Z_LIMIT Standardfehlern liegen."""
    ok = sum(1 for r in rows if r["capped"] == 0 and r["z"] is not None and abs(r["z"]) <= Z_LIMIT)
    capped = sum(r["capped"] for r in rows)
    worst = max((abs(r["z"]) for r in rows if r["z"] is not None), default=float("nan"))
    text = f"{ok} von {len(rows)} Zellen liegen innerhalb von {de(Z_LIMIT, 0)} Standardfehlern der exakten Zeit (größte Abweichung |z| = {de(worst, 1)}; {sum(r['runs'] for r in rows)} Läufe in allen Zellen)."
    if capped:
        return "warning", text + f" {capped} Läufe erreichten die Schrittgrenze nicht (Mittel verzerrt)."
    return ("success" if ok == len(rows) else "warning"), text


# --- Verdikte der Experimente --------------------------------------------------------------------------------------------------------------------
def spread_percent(values: list[float]) -> float:
    """Größter gegen kleinsten Wert in Prozent über dem kleinsten: (max/min − 1)·100."""
    return (max(values) / min(values) - 1.0) * 100.0


def ring_flatness(rows: list[dict], k: int, n_from: int) -> float | None:
    """Spanne in Prozent (größter gegen kleinsten Mittelwert) der Ringzeit mit k Partnern je Seite für n >= n_from; None bei weniger als zwei Zellen."""
    cells = [r["rings"][k] for r in rows if r["n"] >= n_from and r["rings"].get(k) is not None and r["rings"][k]["mean"] == r["rings"][k]["mean"]]
    if len(cells) < 2:
        return None
    return spread_percent([c["mean"] for c in cells])


def verdict_e1(rows: list[dict], eps: float) -> tuple[str, str]:
    """E1: Vollgraph wächst um Zehnerpotenzen, Ring k = 1 und 2 bleibt flach, k = 3 und 4 fallen bis n = 320 weiter."""
    first, last = rows[0], rows[-1]
    head = f"**ε = {de(eps, 2)}:** im Vollgraph wächst die Zeit von {fmt_log10(first['complete_log10'])} (n = {first['n']}) auf {fmt_log10(last['complete_log10'])} (n = {last['n']})."
    flat1, flat2 = ring_flatness(rows, 1, 20), ring_flatness(rows, 2, 40)
    if flat1 is None or flat2 is None:
        return "info", head + " Die vorgerechneten Ringzellen fehlen."
    parts = [f"Im Ring mit k = 1 ändert sich die Zeit ab n = 20 um höchstens {de(flat1, 1)} %, mit k = 2 ab n = 40 um höchstens {de(flat2, 1)} % (größter gegen kleinsten Mittelwert)."]
    for k in (3, 4):
        c40 = next((r["rings"][k] for r in rows if r["n"] == 40), None)
        c320 = next((r["rings"][k] for r in rows if r["n"] == 320), None)
        if c40 and c320 and c40["mean"] == c40["mean"] and c320["mean"] == c320["mean"]:
            parts.append(f"Bei k = {k} fällt die Zeit von n = 40 auf n = 320 noch auf das {de(c320['mean'] / c40['mean'], 2)}-fache.")
    return "info", head + " " + " ".join(parts) + " Ringe sind also „nicht wachsend in n“, aber nur für kleine k praktisch unabhängig von n."


def verdict_e2(rate_rows: list[dict], exp_rows: list[dict], q: float) -> tuple[str, str]:
    """E2: die gemessene Steigung gegen die Chernoff-Rate und gegen die grobe Erwartung; Exponent in ε gegen thr."""
    worst = max(abs(r["diff"]) for r in rate_rows)
    low, high = min(r["rough_over_measured"] for r in rate_rows), max(r["rough_over_measured"] for r in rate_rows)
    a_low, a_high = min(r["a_over_thr"] for r in exp_rows), max(r["a_over_thr"] for r in exp_rows)
    return "success", (f"**q = {de(q, 2)}:** Die gemessene Steigung weicht in allen {len(rate_rows)} Zeilen höchstens um {de(worst, 4)} von KL(q ‖ ε/2)/ln 10 ab; die grobe Erwartung q·log10(1/ε) ist das {de(low, 2)}- bis {de(high, 2)}-fache "
                       f"der gemessenen Rate. Bei festem n wächst T mit ε^(−a), a liegt bei {de(a_low, 2)} bis {de(a_high, 2)} mal thr (n = {exp_rows[0]['n']} bis {exp_rows[-1]['n']}).")


def first_advantage_end(rows: list[dict]) -> dict | None:
    """Die Zeile mit dem kleinsten k, ab dem der Vorteil höchstens 2 ist (Faktor Vollgraph/Ring <= 2); None, wenn es keine gibt."""
    for r in rows:
        if r["ratio"] == r["ratio"] and r["ratio"] <= 2.0:
            return r
    return None


def sawtooth_steps(rows: list[dict]) -> list[tuple[int, int]]:
    """Paare (k, k + 1), bei denen die Ringzeit mit der größeren Nachbarschaft kleiner wird (die Zeit ist nicht monoton in k)."""
    return [(a["k"], b["k"]) for a, b in zip(rows, rows[1:]) if a["mean"] == a["mean"] and b["mean"] == b["mean"] and b["mean"] < a["mean"]]


def verdict_e3(rows: list[dict], n: int, eps: float) -> tuple[str, str]:
    """E3: Wo endet der Vorteil des Netzes? Faktor bei halber Sicht, erste Zeile mit Faktor <= 2, Zacken in k."""
    if not rows:
        return "info", "Die vorgerechneten Zellen fehlen."
    head = f"**n = {n}, ε = {de(eps, 2)}:** "
    half = min(rows, key=lambda r: abs(r["share"] - 0.5))
    end = first_advantage_end(rows)
    parts = [f"Bei k = {rows[0]['k']} (sieht {de(100 * rows[0]['share'], 0)} % der anderen) ist der Ring um den Faktor {fmt_time(rows[0]['ratio'])} schneller als der Vollgraph, bei etwa halber Sicht (k = {half['k']}, {de(100 * half['share'], 0)} %) noch um den Faktor {fmt_time(half['ratio'])}."]
    if end is not None:
        parts.append(f"Der Vorteil sinkt erst bei k = {end['k']} (sieht {de(100 * end['share'], 0)} %) auf höchstens das Doppelte ({de(end['ratio'], 2)}).")
    last = rows[-1]
    if last["ratio"] == last["ratio"] and last["ratio"] < 1:
        parts.append(f"Bei k = {last['k']} (sieht {de(100 * last['share'], 0)} %) ist der Ring langsamer als der Vollgraph (Faktor {de(last['ratio'], 2)}).")
    saw = sawtooth_steps(rows)
    if saw:
        parts.append("Die Zeit ist nicht monoton in k: bei " + ", ".join(f"k = {a} → {b}" for a, b in saw) + " fällt sie, weil die ganzzahlige Schwelle nicht mitwächst (E4).")
    text = head + " ".join(parts)
    capped = sum(r["capped"] for r in rows)
    if capped:
        return "warning", text + f" {capped} Läufe erreichten die Schrittgrenze nicht (Mittel dort nach unten verzerrt, offene Marker)."
    return "info", text


def spread_zero_factors(rows: list[dict]) -> list[tuple[int, int, float]]:
    """Für Zellen mit Spielraum 0 (k = thr): (thr, nächstes thr, Faktor der Zeit) zwischen aufeinanderfolgenden Schwellen."""
    zero = sorted((r for r in rows if r["spread"] == 0 and r["mean"] is not None), key=lambda r: r["thr"])
    seen, unique = set(), []
    for r in zero:
        if r["thr"] not in seen:
            seen.add(r["thr"])
            unique.append(r)
    return [(a["thr"], b["thr"], b["mean"] / a["mean"]) for a, b in zip(unique, unique[1:])]


def verdict_e4(rows: list[dict]) -> tuple[str, str]:
    """E4: Die Zeit hängt an thr und k, nicht stetig an q; sie fällt mit dem Spielraum; Faktor je Schwellenstufe bei Spielraum 0."""
    if not any(r["mean"] is not None for r in rows):
        return "info", "Die vorgerechneten Zellen fehlen."
    factors = spread_zero_factors(rows)
    unique4, seen = [], set()
    for r in sorted((r for r in rows if r["thr"] == 4 and r["mean"] is not None), key=lambda r: r["spread"]):
        if r["spread"] not in seen:
            seen.add(r["spread"])
            unique4.append(r)
    parts = ["Die Zeit ändert sich nur, wenn q die ganzzahlige Schwelle thr ändert (Zellen mit gleichem k und thr sind zahlengleich: gleiche Zufallsströme, kein Messfehler)."]
    if unique4:
        parts.append("Bei thr = 4 fällt sie mit dem Spielraum s = k − thr: " + ", ".join(f"s = {r['spread']}: {fmt_time(r['mean'])}" for r in unique4) + ".")
    if factors:
        parts.append("Bei Spielraum 0 wächst sie je zusätzlicher Schwelle um den Faktor " + ", ".join(de(f, 1) for _, _, f in factors) + ".")
    return "info", " ".join(parts)


def grid_flatness(rows: list[dict]) -> list[dict]:
    """Je Zelle (Nachbarschaft, q, ε): relative Änderung der Zeit von L = 12 auf L = 24 in Prozent (positiv: größer) und die Zahl der abgebrochenen Läufe in beiden Zellen."""
    out = []
    for kind, q, eps in sorted({(r["kind"], r["q"], r["eps"]) for r in rows}):
        by_side = {r["side"]: r for r in rows if r["kind"] == kind and r["q"] == q and abs(r["eps"] - eps) < 1e-12}
        a, b = by_side.get(12), by_side.get(24)
        if a and b and a["mean"] is not None and b["mean"] is not None:
            out.append({"kind": kind, "q": q, "eps": eps, "thr": a["thr"], "edge": a["edge"], "change": (b["mean"] / a["mean"] - 1.0) * 100.0, "capped": a["capped"] + b["capped"]})
    return out


def verdict_e5(rows: list[dict]) -> tuple[str, str]:
    """E5: Gitter: Zeit praktisch unabhängig von L ab L = 12 bei Schwelle höchstens 3; Gitter mit 8 Nachbarn und Schwelle 4 ist nicht gesättigt."""
    flat = grid_flatness(rows)
    if not flat:
        return "info", "Die vorgerechneten Zellen fehlen."
    stable = [f for f in flat if f["thr"] <= 3 and f["capped"] == 0]
    unsat = [f for f in flat if f["thr"] > 3 or f["capped"] > 0]
    worst = max(abs(f["change"]) for f in stable) if stable else float("nan")
    text = f"In {len(stable)} Zellen (Schwelle höchstens 3) ändert sich die Zeit von L = 12 auf L = 24 um höchstens {de(worst, 1)} %: Das Gitter wächst nicht mit n."
    if unsat:
        text += (f" Gitter mit 8 Nachbarn und Schwelle 4 (Randsicht 3) ist nicht gesättigt: {sum(f['capped'] for f in unsat)} Läufe in den beiden größten Gittern brachen an der Schrittgrenze ab; die Zeit ändert sich von L = 12 auf L = 24 um "
                 + " bzw. ".join(f"{de(f['change'], 0)} %" for f in unsat) + " und wird nicht gedeutet.")
        return "warning", text
    return "info", text


def verdict_e6(rows: list[dict], pair: dict, q: float) -> tuple[str, str]:
    """E6: Der Rückwechsel braucht rund das Doppelte im Exponenten; Steigungen gegen KL(q ‖ ε/2) und KL(1 − q ‖ ε/2)."""
    biggest = max(r["n"] for r in rows)
    ratios = {r["eps"]: r["ratio"] for r in rows if r["n"] == biggest}
    ratio_text = ", ".join(f"ε = {de(e, 2)}: {de(v, 2)}" for e, v in sorted(ratios.items(), reverse=True))
    return "info", (f"**q = {de(q, 2)}, ε = 0,1, n = 320 → 640:** Steigung hin {de(pair['slope_hin'], 4)} (KL(q ‖ ε/2)/ln 10 = {de(pair['rate_hin'], 4)}), zurück {de(pair['slope_back'], 4)} "
                    f"(KL(1 − q ‖ ε/2)/ln 10 = {de(pair['rate_back'], 4)}). Verhältnis der Exponenten zurück/hin bei n = {biggest}: {ratio_text}. "
                    "Die Konvention, die gerade gilt, ist also noch schwerer zu verlassen als die alte zu verlassen war.")


def verdict_e7(rows: list[dict]) -> tuple[str, str]:
    """E7: bis wohin das lineare System stimmt, ab wann es um mehr als 1 % abweicht und ob es negativ wird (hängt an Routine und Plattform)."""
    good = [r for r in rows if r["deviation"] < 1e-6]
    bad = [r for r in rows if r["deviation"] >= 1e-2]
    negative = [r for r in rows if r["negative"]]
    if good:
        text = f"Bis n = {max(r['n'] for r in good)} stimmt das lineare System mit dem Logarithmus auf relativ {max(r['deviation'] for r in good):.1e} überein.".replace(".", ",", 1)
    else:
        text = "Das lineare System weicht schon bei kleinem n ab."
    if bad:
        text += f" Ab n = {min(r['n'] for r in bad)} weicht es um mehr als 1 % ab"
        text += f", bei n = {min(r['n'] for r in negative)} liefert es eine negative Zeit." if negative else "."
    text += " Wo es bricht, hängt an der Lösungsroutine und der Plattform; der Logarithmus-Weg bleibt bis n = 20 000 in Millisekunden gültig."
    return ("warning" if bad else "info"), text
