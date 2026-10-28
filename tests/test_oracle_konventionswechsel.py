"""Orakel-Tests (ORACLE-PLAYBOOK §2, Spieltheorie/Markov-Ketten): die erwartete Wechselzeit auf anderen Rechenwegen.

Orakel (jeweils ein ANDERER Weg als der Kern):
  O1  Handrechnung für n = 3 (Vollgraph) und den Ring n = 4, gegen alle drei Wege des Kerns (Logarithmus, lineares System, 2^n-Zustandsraum);
  O2  exakte Bruchrechnung (`fractions`, Gauß-Elimination) für den Ring n = 4 bis 6;
  O3  Logarithmus-Weg gegen das lineare System (n <= 40) und gegen eine eigene Aufzählung aller Belegungen mit Schleifen und scipy.sparse (n <= 12; Vollgraph, Ring, Torus 3 × 4);
  O4  der Rückwechsel (gespiegelte Kette) gegen die UNgespiegelte Dynamik im 2^n-Raum (Start „alle A“, Ziel thr_B + 2 B-Betriebe);
  O5  exakte Zeit gegen die Simulation (feste Zufalls-Seeds, Marge 4 Standardfehler; ein Mittel aus wenigen Läufen genügt nicht für „z unter 2“);
  O6  Steigung gegen die Chernoff-Rate KL(q ‖ ε/2)/ln 10 und die Binomial-Eigenschaft der Gewichte unterhalb der Schwelle."""

import math
from fractions import Fraction

import numpy as np
import pytest

import kvw_exact as E
import kvw_graphs as GR
import kvw_model as M
import kvw_sim as SI

sp = pytest.importorskip("scipy.sparse")
spla = pytest.importorskip("scipy.sparse.linalg")


# --- Orakel: Aufzählung aller Belegungen mit einfachen Schleifen --------------------------------------------------------------------------------
def oracle_hitting(nb, thr, eps, start, stop):
    """Erwartete Schritte je Betrieb von der Belegung `start` (Ganzzahl, Bit i = Betrieb i führt A), bis `stop(state)` gilt. Eigene Aufzählung: jede Belegung, jeder gezogene Betrieb,
    jede Antwort; (I − P) t = 1 über die Zustände, in denen nicht gestoppt wird."""
    n = len(nb)
    transient = [s for s in range(1 << n) if not stop(s)]
    index = {s: i for i, s in enumerate(transient)}
    rows, cols, vals = [], [], []
    for s in transient:
        row = index[s]
        diag = 1.0
        for i in range(n):
            seen_a = sum((s >> int(j)) & 1 for j in nb[i])
            p_a = (1.0 - eps) * (1.0 if seen_a >= thr else 0.0) + eps / 2.0
            for new_bit, prob in ((1, p_a), (0, 1.0 - p_a)):
                t = (s | (1 << i)) if new_bit else (s & ~(1 << i))
                if t in index:
                    if t == s:
                        diag -= prob / n
                    else:
                        rows.append(row)
                        cols.append(index[t])
                        vals.append(-prob / n)
                else:
                    pass  # absorbiert: kein Eintrag
        rows.append(row)
        cols.append(row)
        vals.append(diag)
    a = sp.csr_matrix((vals, (rows, cols)), shape=(len(transient), len(transient)))
    t = spla.spsolve(a.tocsc(), np.ones(len(transient)))
    return float(t[index[start]] / n)


def popcount(x):
    return bin(x).count("1")


def oracle_forward(nb, thr, eps):
    n = len(nb)
    goal = (n + 1) // 2
    return oracle_hitting(nb, thr, eps, 0, lambda s: popcount(s) >= goal)


def fraction_forward(nb, thr, eps):
    """Exakte Bruchrechnung (Gauß-Elimination mit `Fraction`) derselben erwarteten Zeit, für n <= 6."""
    n = len(nb)
    goal = (n + 1) // 2
    transient = [s for s in range(1 << n) if popcount(s) < goal]
    index = {s: i for i, s in enumerate(transient)}
    m = len(transient)
    mat = [[Fraction(0)] * (m + 1) for _ in range(m)]
    eps = Fraction(eps).limit_denominator(1000)
    for s in transient:
        r = index[s]
        mat[r][r] += 1
        mat[r][m] = Fraction(1)
        for i in range(n):
            seen_a = sum((s >> int(j)) & 1 for j in nb[i])
            p_a = (1 - eps) * (1 if seen_a >= thr else 0) + eps / 2
            for bit, prob in ((1, p_a), (0, 1 - p_a)):
                t = (s | (1 << i)) if bit else (s & ~(1 << i))
                if t in index:
                    mat[r][index[t]] -= prob / n
    for c in range(m):
        piv = next(r for r in range(c, m) if mat[r][c] != 0)
        mat[c], mat[piv] = mat[piv], mat[c]
        inv = 1 / mat[c][c]
        mat[c] = [v * inv for v in mat[c]]
        for r in range(m):
            if r != c and mat[r][c] != 0:
                f = mat[r][c]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[c])]
    return mat[index[0]][m] / n


# --- O1: Handrechnung ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.2, 0.1])
def test_o1_three_firms_by_hand_against_all_three_ways(mini3, eps):
    want = mini3["time"][eps]
    thr = M.threshold(mini3["q"], mini3["n"] - 1)
    assert thr == mini3["thr"]
    assert math.exp(E.complete_log_time(3, thr, eps)) == pytest.approx(want, rel=1e-12)
    assert E.complete_linear_time(3, thr, eps) == pytest.approx(want, rel=1e-12)
    assert E.graph_exact_time(GR.complete_nb(3), thr, eps) == pytest.approx(want, rel=1e-12)
    assert oracle_forward(GR.complete_nb(3), thr, eps) == pytest.approx(want, rel=1e-9)


def test_o1_hand_values_are_the_stated_numbers(mini3):
    """Die Handrechnung der Fixture ergibt die im Plan und in der README genannten Zahlen 5,5556 und 10,5263."""
    assert mini3["time"][0.2] == pytest.approx(5.5556, abs=5e-5) and mini3["time"][0.1] == pytest.approx(10.5263, abs=5e-5)


def test_o1_ring_of_four_by_hand(ring4):
    nb = GR.ring_nb(ring4["n"], ring4["k"])
    thr = M.threshold(ring4["q"], 2)
    assert thr == ring4["thr"]
    assert ring4["time"] == pytest.approx(4.2105, abs=5e-5)
    assert E.graph_exact_time(nb, thr, ring4["eps"]) == pytest.approx(ring4["time"], rel=1e-12)
    assert oracle_forward(nb, thr, ring4["eps"]) == pytest.approx(ring4["time"], rel=1e-9)


# --- O2: exakte Bruchrechnung -------------------------------------------------------------------------------------------------------------------
def test_o2_ring_of_four_is_the_exact_fraction(ring4):
    """4,2105 ist der Bruch (10 + 3,25/0,475)/4 = 3,25/0,475: mit ε = 1/5 exakt 1 000/…: Bruchrechnung statt Gleitkomma."""
    nb = GR.ring_nb(4, 1)
    value = fraction_forward(nb, 1, 0.2)
    assert value == Fraction(10, 4) + Fraction(325, 100) / Fraction(475, 1000) / 4
    assert float(value) == pytest.approx(E.graph_exact_time(nb, 1, 0.2), rel=1e-12)


@pytest.mark.parametrize("name,nb,thr", [("Vollgraph 5", GR.complete_nb(5), 2), ("Ring 5, k = 1", GR.ring_nb(5, 1), 1), ("Ring 6, k = 2", GR.ring_nb(6, 2), 2), ("Vollgraph 6", GR.complete_nb(6), 2)])
def test_o2_fractions_agree_with_the_kernel_to_rounding(name, nb, thr):
    value = fraction_forward(nb, thr, 0.3)
    assert float(value) == pytest.approx(E.graph_exact_time(nb, thr, 0.3), rel=1e-11), name


# --- O3: drei Wege --------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 15, 20])
@pytest.mark.parametrize("q", [0.2, 0.4, 0.45])
@pytest.mark.parametrize("eps", [0.3, 0.1])
def test_o3_log_equals_linear_system_for_small_n(n, q, eps):
    thr = M.threshold(q, n - 1)
    assert math.exp(E.complete_log_time(n, thr, eps)) == pytest.approx(E.complete_linear_time(n, thr, eps), rel=1e-9)


@pytest.mark.parametrize("n,eps", [(30, 0.3), (40, 0.3), (30, 0.1)])
def test_o3_log_equals_linear_system_up_to_the_safe_limit(n, eps):
    """Bis n = 40 (das Limit des linearen Systems) auf 1e-5; bei ε = 0,1 und n = 30 sind es noch Abweichungen unter 1e-6 (Rundungsfehler wachsen mit der Größe der Zeit)."""
    thr = M.threshold(0.4, n - 1)
    assert math.exp(E.complete_log_time(n, thr, eps)) == pytest.approx(E.complete_linear_time(n, thr, eps), rel=1e-5)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("eps", [0.3, 0.1])
def test_o3_log_equals_state_space_enumeration_on_complete_graphs(n, eps):
    thr = M.threshold(0.4, n - 1)
    nb = GR.complete_nb(n)
    assert math.exp(E.complete_log_time(n, thr, eps)) == pytest.approx(oracle_forward(nb, thr, eps), rel=1e-9)
    assert E.graph_exact_time(nb, thr, eps) == pytest.approx(oracle_forward(nb, thr, eps), rel=1e-9)


@pytest.mark.parametrize("n,k", [(5, 1), (7, 2), (8, 1), (9, 3), (10, 2), (12, 3)])
def test_o3_dense_kernel_equals_enumeration_on_rings(n, k):
    thr = M.threshold(0.4, 2 * k)
    nb = GR.ring_nb(n, k)
    assert E.graph_exact_time(nb, thr, 0.2) == pytest.approx(oracle_forward(nb, thr, 0.2), rel=1e-9)


@pytest.mark.parametrize("lx,ly,kind", [(3, 3, "vn"), (3, 3, "moore"), (3, 4, "vn"), (3, 4, "moore")])
def test_o3_dense_kernel_equals_enumeration_on_tori(lx, ly, kind):
    nb = GR.torus_nb(lx, ly, kind)
    thr = M.threshold(0.4, nb.shape[1])
    assert E.graph_exact_time(nb, thr, 0.3) == pytest.approx(oracle_forward(nb, thr, 0.3), rel=1e-9)


def test_o3_complete_graph_of_twelve_matches_the_log_way():
    """Der größte live gerechnete Zustandsraum (2^12): Vollgraph über alle Belegungen gleich dem Logarithmus-Weg."""
    thr = M.threshold(0.4, 11)
    assert E.graph_exact_time(GR.complete_nb(12), thr, 0.3) == pytest.approx(math.exp(E.complete_log_time(12, thr, 0.3)), rel=1e-9)


# --- O4: Rückwechsel --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [5, 6, 8, 9])
@pytest.mark.parametrize("eps", [0.3, 0.2])
def test_o4_mirrored_chain_equals_unmirrored_dynamics(n, eps):
    """Rückwechsel: Die ungespiegelte Dynamik (A bei mindestens thr A-Partnern) von „alle A“ bis mindestens thr_B + 2 B-Betrieben, gezählt im 2^n-Raum, ist gleich der gespiegelten Kette im Kern."""
    thr = M.threshold(0.4, n - 1)
    thr_back, target_back = E.mirror(n, thr)
    assert thr_back == n - thr and target_back == min(n, thr_back + 2)
    nb = GR.complete_nb(n)
    full = (1 << n) - 1
    unmirrored = oracle_hitting(nb, thr, eps, full, lambda s: n - popcount(s) >= target_back)
    assert math.exp(E.mirror_log_time(n, thr, eps)) == pytest.approx(unmirrored, rel=1e-9)


# --- O5: Simulation gegen exakt --------------------------------------------------------------------------------------------------------------
def z_score(times, exact):
    mean, se = SI.mean_and_error(times)
    return (mean - exact) / se


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_o5_complete_graph_simulation_within_four_standard_errors(seed):
    n, eps = 8, 0.3
    thr = M.threshold(0.4, n - 1)
    exact = math.exp(E.complete_log_time(n, thr, eps))
    res = SI.simulate_complete(n, thr, eps, 3000, seed, 200_000)
    assert res.capped == 0 and abs(z_score(res.times, exact)) < 4


@pytest.mark.parametrize("seed", [21, 22])
def test_o5_agent_level_simulation_on_complete_graph_within_four_standard_errors(seed):
    n, eps = 8, 0.3
    thr = M.threshold(0.4, n - 1)
    exact = math.exp(E.complete_log_time(n, thr, eps))
    res = SI.simulate(GR.complete_nb(n), thr, eps, 3000, seed, 200_000)
    assert res.capped == 0 and abs(z_score(res.times, exact)) < 4


@pytest.mark.parametrize("name,nb", [("Ring 10, k = 2", GR.ring_nb(10, 2)), ("Ring 9, k = 1", GR.ring_nb(9, 1)), ("Torus 3×3, 4 Nachbarn", GR.torus_nb(3, 3, "vn")), ("Torus 3×4, 8 Nachbarn", GR.torus_nb(3, 4, "moore"))])
def test_o5_networks_within_four_standard_errors(name, nb):
    thr = M.threshold(0.4, nb.shape[1])
    exact = E.graph_exact_time(nb, thr, 0.3)
    res = SI.simulate(nb, thr, 0.3, 3000, 31, 300_000)
    assert res.capped == 0 and abs(z_score(res.times, exact)) < 4, name


def test_o5_mirrored_simulation_within_four_standard_errors():
    n, eps = 10, 0.3
    thr = M.threshold(0.4, n - 1)
    thr_back, target_back = E.mirror(n, thr)
    exact = math.exp(E.mirror_log_time(n, thr, eps))
    res = SI.simulate_complete(n, thr_back, eps, 3000, 41, 400_000, target_back)
    assert res.capped == 0 and abs(z_score(res.times, exact)) < 4


# --- O6: Rate und Binomial-Eigenschaft ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [0.2, 0.1, 0.05, 0.02])
def test_o6_slope_equals_chernoff_rate(eps):
    """Steigung von log10 T je Betrieb zwischen n = 800 und 1 600 gegen KL(0,4 ‖ ε/2)/ln 10, Marge 0,002 (gemessen: Abweichung unter 0,0003)."""
    def log10_t(n):
        return E.complete_log10_time(n, M.threshold(0.4, n - 1), eps)
    slope = (log10_t(1600) - log10_t(800)) / 800
    assert slope == pytest.approx(E.chernoff_rate(0.4, eps), abs=0.002)


def test_o6_weights_below_the_threshold_are_binomial():
    """Für k < thr ist up[k]/down[k+1] = (n − k)/(k + 1) · (ε/2)/(1 − ε/2): das Verhältnis der Gewichte von Bin(n, ε/2)."""
    n, eps = 40, 0.1
    thr = M.threshold(0.4, n - 1)
    up, down = E.birth_death_rates(n, thr, eps)
    for k in range(thr):
        assert up[k] / down[k + 1] == pytest.approx((n - k) / (k + 1) * (eps / 2) / (1 - eps / 2), rel=1e-12)
    p = eps / 2
    log_weights = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p) for k in range(thr + 1)]
    chain = np.concatenate([[0.0], np.cumsum(np.log(up[:thr]) - np.log(down[1: thr + 1]))])
    assert np.allclose(np.array(log_weights) - log_weights[0], chain, atol=1e-9)


def test_o6_rate_is_not_the_rough_expectation():
    """Die grobe Erwartung q·log10(1/ε) liegt über der gemessenen Steigung (1,3- bis 2,1-fach in den vier ε der Tabelle)."""
    for eps in (0.2, 0.1, 0.05, 0.02):
        measured = (E.complete_log10_time(1600, M.threshold(0.4, 1599), eps) - E.complete_log10_time(800, M.threshold(0.4, 799), eps)) / 800
        assert 1.3 < E.rough_rate(0.4, eps) / measured < 2.1
