"""Wie lange dauert der Wechsel einer Konvention? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Reihe Koordinationsspiele. Eine bessere Konvention (neues Format A, risikodominant) verdrängt die alte (B) nur durch Fehler; die Demo rechnet exakt, wie lange das im
Erwartungswert dauert, woher die Rate der exponentiellen Zunahme kommt, und wo ein Netz mit wenigen Partnern den Wechsel um viele Zehnerpotenzen verkürzt.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import kvw_constants as C
import kvw_evaluation as V
import kvw_exact as E
import kvw_experiments as X
import kvw_graphs as GR
import kvw_grid as GRID
import kvw_scenario as SC
import kvw_trace as TR
import kvw_visualization as Z
from kvw_layout import dunkles_design, kopf_css
from kvw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Wie lange dauert der Wechsel einer Konvention? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _main(network, n, k, q, eps, direction, budget):
    return V.analyse_main(V.Settings(network=network, n=n, k=k, q=q, eps=eps, direction=direction, budget=budget))


@st.cache_data(show_spinner=False)
def _exact_small(kind, n, k, thr, eps):
    spec = SC.NetworkSpec(kind, n, k, GR.degree(kind, n, k), thr, GR.edge_view(kind, k))
    return V.exact_small_log10(spec, eps)


@st.cache_data(show_spinner=False)
def _trace(kind, n, k, thr, eps, seed, cap, target):
    return TR.trace_run(kind, n, k, thr, eps, seed, cap, target)


@st.cache_data(show_spinner=False)
def _frames(kind, n, k, thr, eps, seed, steps):
    return TR.frames(kind, n, k, thr, eps, seed, list(steps))


@st.cache_data(show_spinner=False)
def _simulate(network, n, k, q, eps, direction, runs, budget, seed, exact_small):
    view = V.analyse_main(V.Settings(network=network, n=n, k=k, q=q, eps=eps, direction=direction, runs=runs, budget=budget, seed=seed))
    return V.run_main_simulation(view, exact_small)


@st.cache_data(show_spinner=False)
def _check(q, eps, runs, budget, seed):
    return V.run_check(q, eps, runs, seed, budget)


@st.cache_data(show_spinner=False)
def _measurements():
    return X.load_measurements()


@st.cache_data(show_spinner=False)
def _e2(q):
    return X.e2_rate_rows(q), X.e2_exponent_rows(q)


@st.cache_data(show_spinner=False)
def _e6(q):
    return X.e6_rows(q), X.e6_rate_pair(q, 0.1, 320, 640)


@st.cache_data(show_spinner=False)
def _e7():
    return X.e7_rows()


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def cell_text(cell):
    """Eine vorgerechnete Zelle für die Tabelle: Mittel ± Standardfehler; mit Sternchen, wenn Läufe abgebrochen sind (Mittel dann nach unten verzerrt)."""
    if cell is None or cell["mean"] is None or cell["mean"] != cell["mean"]:
        return "–"
    star = "*" if cell["capped"] else ""
    return f"{V.fmt_time(cell['mean'])} ± {V.fmt_time(cell['se'])}{star}"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kvw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⏳ Wie lange dauert der Wechsel einer Konvention?")
st.markdown(
    """
**Drittes Stück der Reihe Koordinationsspiele.** Das erste Stück (`koordinationsspiel-demo`) zeigt die Gleichgewichte eines Koordinationsspiels, das zweite (`gleichgewichtsauswahl-demo`), welche Konvention auf Dauer gewählt wird;
hier geht es um die Frage danach: Betriebe führen das **neue Palettenformat A** oder das **alte B**, und A lohnt sich für einen Betrieb, sobald mindestens der Anteil **q** seiner Partner es führt (q < 1/2: A ist risikodominant).
**Wie lange braucht es, bis A das alte Format B verdrängt, wenn nur Fehler es anstoßen, und warum wächst die Zeit bei „jeder trifft jeden“ um Zehnerpotenzen mit der Zahl der Betriebe, in einem Nachbarschaftsnetz mit wenigen Partnern aber nicht?**

**Eigennutz und die Lernregel (beste Antwort, mit Wahrscheinlichkeit ε eine Zufallswahl) sind Modellannahmen**, keine Aussagen über Menschen oder Terminals. Jede Zeit ist ein Erwartungswert mit Streuung, keine Prognose.

Weiter unten: sieben Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Wechsel", expanded=True):
    st.markdown(
        r"""
1. **Beste Antwort mit Fehlern.** Ein gleichverteilt gezogener Betrieb mit $d$ Partnern führt A, wenn mindestens $\mathrm{thr} = \lceil q\,d\rceil$ Partner A führen (bei Gleichstand gewinnt A); mit Wahrscheinlichkeit $\varepsilon$ wählt er stattdessen A oder B je zur Hälfte.
Start: alle führen B. **Ziel:** mindestens die Hälfte führt A. **Zeit:** Aktualisierungen je Betrieb (Einzelaktualisierungen geteilt durch $n$).

2. **Warum die Zeit wächst.** Solange weniger als $\mathrm{thr}$ Partner A führen, geht jeder Betrieb zu B zurück. A wächst nur durch Fehler, und zwar von selbst erst, wenn $\mathrm{thr}$ Betriebe gleichzeitig A führen. Im Vollgraph (jeder trifft jeden) sind das $\mathrm{thr} \approx q\,n$ Fehler auf einmal; die Zahl der A-Betriebe unterhalb der Schwelle ist dort binomial verteilt:

$$k \sim \mathrm{Bin}\!\left(n, \tfrac{\varepsilon}{2}\right)$$

3. **Die Rate.** Die Wartezeit ist ungefähr der Kehrwert der Wahrscheinlichkeit, dass $\mathrm{thr}$ Betriebe zugleich A führen. Je Betrieb wächst $\log_{10} T$ daher um die relative Entropie (Chernoff-Rate), nicht um $q\log_{10}(1/\varepsilon)$ (Experiment E2):

$$\frac{\mathrm{d}\,\log_{10} T}{\mathrm{d}n} \to \frac{\mathrm{KL}\!\left(q \,\Vert\, \tfrac{\varepsilon}{2}\right)}{\ln 10}, \qquad \mathrm{KL}(q\,\Vert\,p) = q\ln\frac{q}{p} + (1-q)\ln\frac{1-q}{1-p}$$

4. **Im Netz.** Sieht ein Betrieb nur wenige Partner, ist die nötige Zahl gleichzeitiger Fehler $\mathrm{thr}$ klein, und rückt der Rand eines A-Blocks von selbst vor, wächst der Block weiter. Besonders kurz ist die Zeit, wenn eine gerade Kante ohne Fehler vorankommt, die Randsicht also mindestens $\mathrm{thr}$ beträgt (Spielraum $s \ge 0$). Braucht die Kante Fehler ($s < 0$), kann die Zeit trotzdem klein sein (4 Nachbarn, Schwelle 2) oder das Gitter nicht gesättigt (8 Nachbarn, Schwelle 4, Experiment E5).
        """
    )

with st.expander("Erklärtabelle: die Begriffe"):
    st.markdown(
        """
| Begriff | Bedeutung | Wo gezeigt |
|---|---|---|
| **Schwelle q, thr** | Anteil bzw. Zahl der Partner, ab der das neue Format A lohnt: thr = ⌈q·d⌉ bei d Partnern. | Seitenleiste, Hauptansicht |
| **Fehlerrate ε** | Mit dieser Wahrscheinlichkeit wählt ein Betrieb zufällig (A oder B je zur Hälfte) statt der besten Antwort. | Seitenleiste |
| **Wechselzeit** | Erwartete Aktualisierungen je Betrieb von „alle B“ bis mindestens die Hälfte A. Ein Erwartungswert, keine Prognose. | Hauptansicht, alle Tabs |
| **Randsicht, Spielraum s** | Wie viele A-Partner ein Betrieb am geraden Rand eines A-Blocks sieht (Ring: k; Gitter mit 4 Nachbarn: 1; mit 8: 3), minus thr. Bei s ≥ 0 rückt die Kante ohne Fehler vor. | Hauptansicht, E4, E5 |
| **Rate** | Zuwachs von log10 der Wechselzeit je zusätzlichem Betrieb im Vollgraph. | E2 |
| **Rückwechsel** | Die Gegenrichtung von „alle A“ zurück zu B, mit gespiegelter Schwelle. | E6 |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🕸️ Wer sieht wen?")
    network = st.selectbox("Netz", C.NETWORKS, key="network_select",
                           help="Vollgraph: jeder trifft jeden. Ring: jeder sieht die k Nachbarn links und rechts. Gitter: Torus mit 4 oder 8 Nachbarn.")
    n = st.slider("Betriebe n", *bounds("n_slider"), key="n_slider", help="Zahl der Betriebe. Beim Gitter wird n auf die nächste Quadratzahl L² gerundet (L von 3 bis 31).")
    is_ring = C.NETWORK_KIND[network] == "ring"
    k = st.slider("Partner je Seite k (nur Ring)", *bounds("k_slider"), key="k_slider", disabled=not is_ring,
                  help="Nur im Ring wirksam: jeder Betrieb sieht k Nachbarn links und k rechts (d = 2k). Auf (n − 1)/2 begrenzt; in den anderen Netzen ausgegraut.")
    st.header("⚖️ Spiel und Fehler")
    q = st.slider("Schwelle q", *bounds("q_slider"), key="q_slider", step=0.01,
                  help="Anteil der Partner, ab dem das neue Format A lohnt (q < 1/2: A ist risikodominant). Die Zahl der nötigen Partner ist thr = ⌈q·d⌉.")
    eps = st.slider("Fehlerrate ε", *bounds("eps_slider"), key="eps_slider", step=0.01, help="Mit dieser Wahrscheinlichkeit wählt ein Betrieb zufällig (A oder B je zur Hälfte) statt der besten Antwort.")
    is_complete = C.NETWORK_KIND[network] == "complete"
    direction = st.selectbox("Richtung (nur Vollgraph)", C.DIRECTIONS, key="direction_select", disabled=not is_complete,
                             help="Hinwechsel: von „alle B“ zum neuen Format A. Rückwechsel: von „alle A“ zurück zu B mit gespiegelter Schwelle. Nur im Vollgraph gerechnet; in den anderen Netzen ausgegraut.")
    st.header("🖥️ Simulation")
    runs = st.slider("Läufe", *bounds("runs_slider"), key="runs_slider", help="Wie viele Läufe gleichzeitig simuliert werden (jeder mit eigenem Zufallsstrom).")
    budget = st.select_slider("Schrittgrenze je Lauf", options=C.SIM_BUDGETS, key="budget_select",
                              help="So viele Einzelaktualisierungen darf jeder Lauf höchstens rechnen. Läufe, die das Ziel bis dahin nicht erreichen, werden gemeldet und nie gemittelt.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"network_select": network, "n_slider": int(n), "k_slider": int(k), "q_slider": float(q), "eps_slider": float(eps), "direction_select": direction,
                   "runs_slider": int(runs), "budget_select": int(budget), "seed_input": int(seed)})

settings = V.Settings(network=network, n=int(n), k=int(k), q=float(q), eps=float(eps), direction=direction, runs=int(runs), budget=int(budget), seed=int(seed))
view = _main(settings.network, settings.n, settings.k, settings.q, settings.eps, settings.direction, settings.budget)
spec = view.spec
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Wie lange dauert der Wechsel?")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Betriebe n", str(spec.n))
m2.metric("Partner je Betrieb d", str(spec.d))
m3.metric("Nötig: thr", str(view.thr), help="Zahl der Partner, die schon das Ziel-Format führen müssen, damit es sich für den einzelnen Betrieb lohnt.")
m4.metric("Randsicht / Spielraum s", "–" if spec.edge is None else f"{spec.edge} / {spec.spread}", help="A-Partner am geraden Rand eines A-Blocks und Randsicht minus thr.")
exact_small = None
if view.log10_time is not None:
    m5.metric("Erwartete Zeit (exakt)", V.fmt_log10(view.log10_time), help="Aktualisierungen je Betrieb; Erwartungswert, keine Prognose.")
elif spec.n <= E.EXACT_MAX_N:
    with st.spinner("Rechne exakt über alle 2^n Belegungen …"):
        exact_small = _exact_small(spec.kind, spec.n, spec.k, spec.thr, settings.eps)
    m5.metric("Erwartete Zeit (exakt)", V.fmt_log10(exact_small), help="Über alle 2^n Belegungen gerechnet; Aktualisierungen je Betrieb.")
else:
    m5.metric("Erwartete Zeit", "Simulation", help=f"Kein Vollgraph und n > {E.EXACT_MAX_N}: die Zeit steht in der Simulation.")
if spec.kind == "ring" and spec.k != settings.k:
    st.caption(f"k wurde auf {spec.k} begrenzt, weil der Ring n ≥ 2k + 1 Betriebe braucht.")
if spec.kind in ("vn", "moore") and spec.n != settings.n:
    st.caption(f"Das Gitter hat n = L² Betriebe: n = {settings.n} wurde auf {spec.n} (L = {GR.side_of(spec.kind, spec.n)}) gerundet.")

st.markdown("#### ❓ Kernfrage: Wie lange dauert der Wechsel, und kommt der Rand voran?")
v_main = V.verdict_main(view)
show(v_main)
verdicts.append(v_main)
v_spread = V.verdict_spread(spec)
show(v_spread)
verdicts.append(v_spread)

st.markdown("**Ein Lauf als Bild** (Lauf 0 des Zufalls-Seeds)")
target_arg = view.target if view.back else None
tr = _trace(spec.kind, spec.n, spec.k, view.thr, settings.eps, settings.seed, C.TRACE_CAP, target_arg)
if spec.kind == "complete":
    st.plotly_chart(Z.build_trace(tr.series, spec.n, tr.target, tr.reached, "Anteil der Betriebe mit " + ("B" if view.back else "A")), width="stretch", key="trace_chart")
else:
    steps = tuple(TR.frame_steps(len(tr.series), C.SNAPSHOT_COUNT))
    frame_list = _frames(spec.kind, spec.n, spec.k, view.thr, settings.eps, settings.seed, steps)
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(Z.build_trace(tr.series, spec.n, tr.target, tr.reached), width="stretch", key="trace_chart")
    with c_right:
        index = st.slider("Zeitpunkt im Verlauf", 0, len(steps) - 1, len(steps) - 1, help="Bild der Belegung nach so vielen Schritten; ganz rechts: das Ende dieses Laufs.")
        state = frame_list[index]
        if spec.kind == "ring":
            st.plotly_chart(Z.build_ring(state), width="stretch", key="frame_chart")
        else:
            st.plotly_chart(Z.build_grid(state, GR.side_of(spec.kind, spec.n)), width="stretch", key="frame_chart")
        st.caption(f"Nach {V.thousands(steps[index])} Schritten ({V.de(steps[index] / spec.n, 1)} Aktualisierungen je Betrieb): {sum(state)} von {spec.n} Betrieben führen A (grün), die übrigen B (orange).")
reached_text = f"Ziel nach {V.thousands(len(tr.series))} Schritten erreicht ({V.fmt_time(tr.time)} Aktualisierungen je Betrieb)." if tr.reached else (
    f"Ziel in diesem Lauf nach {V.thousands(C.TRACE_CAP)} Schritten nicht erreicht: die Wechselzeit liegt über dieser Grenze (die Simulation unten rechnet bis zur eingestellten Schrittgrenze).")
st.caption(reached_text + " Ein einzelner Lauf zeigt, wie ein Wechsel aussieht, nicht, wie lange er typischerweise dauert.")

st.markdown("**Simulation vieler Läufe** (vektorisiert, jeder Lauf mit eigenem Zufallsstrom)")
sim_signature = (settings.network, settings.n, settings.k, settings.q, settings.eps, settings.direction, settings.runs, settings.budget, settings.seed)
if st.button("▶️ Simulation starten", key="sim_button", help=f"Rechnet {settings.runs} Läufe mit je höchstens {V.thousands(settings.budget)} Schritten; höchstens rund {max(2, math.ceil(settings.budget / C.SIM_STEPS_PER_SECOND))} Sekunden."):
    st.session_state["sim_signature"] = sim_signature
if st.session_state.get("sim_signature") == sim_signature:
    with st.spinner(f"Simuliere {settings.runs} Läufe mit höchstens {V.thousands(settings.budget)} Schritten (höchstens rund {max(2, math.ceil(settings.budget / C.SIM_STEPS_PER_SECOND))} Sekunden) …"):
        summary = _simulate(*sim_signature, exact_small=(None if exact_small is None else 10 ** exact_small))
    show(V.verdict_simulation(summary))
    verdicts.append(V.verdict_simulation(summary))

st.markdown("**Formel gegen Simulation** (drei Netze mit je 12 Betrieben, exakt über alle 2^12 Belegungen; Schwelle q wie in der Seitenleiste, Fehlerrate ε = 0,3 fest, damit die Läufe kurz bleiben)")
if st.button("▶️ Formel gegen Simulation rechnen", key="check_button", help=f"Je Netz {C.CHECK_RUNS} Läufe mit höchstens {V.thousands(C.CHECK_CAP)} Schritten; rechnet einige Sekunden."):
    st.session_state["check_done"] = True
if st.session_state.get("check_done"):
    with st.spinner("Rechne exakt über alle 2^12 Belegungen und simuliere (einige Sekunden) …"):
        rows_check = _check(settings.q, C.CHECK_EPS, C.CHECK_RUNS, C.CHECK_CAP, C.CHECK_SEED)
    table([{"Netz": r["label"], "thr": r["thr"], "exakt": V.fmt_time(r["exact"]), "Simulation (Mittel ± Standardfehler)": f"{V.fmt_time(r['mean'])} ± {V.fmt_time(r['se'])}", "z": "–" if r["z"] is None else V.de(r["z"], 1),
            "abgebrochene Läufe": r["capped"]} for r in rows_check])
    v_check = V.verdict_check(rows_check)
    show(v_check)
    verdicts.append(v_check)

measure = _measurements()
if measure is None:
    st.warning("Die vorgerechneten Messreihen (data/measurements.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_measurements.py` ausführen. Die Tabs E1, E3, E4 und E5 zeigen dann nur die exakten Teile.")
cells = [] if measure is None else measure["cells"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["1 Zeit über n", "2 Welche Rate?", "3 Wie groß darf die Nachbarschaft sein?", "4 Die Treppe der Schwelle", "5 Gitter", "6 Rückwechsel", "7 Warum im Logarithmus"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Wie wächst die Zeit mit der Zahl der Betriebe?")
    st.caption(f"Schwelle q = {V.de(GRID.Q_MEASURE, 1)}. Linie: Vollgraph, exakt im Logarithmus gerechnet. Punkte: Ring mit k Partnern je Seite, Simulation (vorgerechnet: {len(GRID.SEEDS)} feste Zufalls-Ströme zu je {GRID.E1_RUNS} Läufen, "
               "Mittel ± Standardfehler; offene Marker: Läufe an der Schrittgrenze abgebrochen, Mittel nach unten verzerrt).")
    e1_eps = st.selectbox("Fehlerrate ε", GRID.E1_EPS, index=1, key="e1_eps", format_func=lambda e: V.de(e, 2))
    rows1 = X.e1_rows(cells, e1_eps)
    st.plotly_chart(Z.build_time_over_n(rows1, e1_eps), width="stretch", key="e1_chart")
    table([{"n": r["n"], "Vollgraph (exakt)": V.fmt_log10(r["complete_log10"]), **{f"Ring k = {k}": cell_text(r["rings"][k]) for k in GRID.E1_KS}} for r in rows1])
    v1 = V.verdict_e1(rows1, e1_eps)
    show(v1)
    verdicts.append(v1)

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Welche Rate?")
    rate_rows, exp_rows = _e2(settings.q)
    st.caption(f"Zuwachs von log10 der Wechselzeit je zusätzlichem Betrieb im Vollgraph, exakt gerechnet zwischen n = {C.RATE_N_LOW} und n = {C.RATE_N_HIGH}; Schwelle q = {V.de(settings.q, 2)} aus der Seitenleiste.")
    st.plotly_chart(Z.build_rates(rate_rows), width="stretch", key="e2_chart")
    table([{"ε": V.de(r["eps"], 2), "gemessen": V.de(r["measured"], 4), "KL(q ‖ ε/2)/ln 10": V.de(r["chernoff"], 4), "q·log10(1/ε)": V.de(r["rough"], 4), "grobe Erwartung / gemessen": V.de(r["rough_over_measured"], 2)} for r in rate_rows])
    st.markdown(f"**Abhängigkeit von ε bei festem n:** T wächst wie ε^(−a) (Steigung von ln T gegen ln(1/ε) zwischen ε = {V.de(C.EXPONENT_EPS[0], 2)} und {V.de(C.EXPONENT_EPS[1], 2)}); a liegt nahe bei der Zahl der nötigen Betriebe thr.")
    table([{"n": r["n"], "thr": r["thr"], "a": V.de(r["a"], 2), "a / thr": V.de(r["a_over_thr"], 3)} for r in exp_rows])
    v2 = V.verdict_e2(rate_rows, exp_rows, settings.q)
    show(v2)
    verdicts.append(v2)

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Wie groß darf die Nachbarschaft sein, bis der Vorteil des Netzes endet?")
    st.caption(f"Ring mit wachsender Nachbarschaft k, Schwelle q = {V.de(GRID.Q_MEASURE, 1)}. Vorteil = exakte Zeit im Vollgraph geteilt durch die simulierte Ringzeit ({len(GRID.SEEDS)} Ströme zu je {GRID.E3_RUNS} Läufen, vorgerechnet).")
    labels3 = {i: f"n = {nn}, ε = {V.de(ee, 2)}" for i, (nn, ee, _) in enumerate(GRID.E3_CONFIGS)}
    pick3 = st.selectbox("Zelle", list(labels3), index=1, format_func=lambda i: labels3[i], key="e3_pick")
    n3, eps3, _ = GRID.E3_CONFIGS[pick3]
    rows3 = X.e3_rows(cells, n3, eps3)
    if rows3:
        st.plotly_chart(Z.build_neighbourhood(rows3, n3, eps3), width="stretch", key="e3_chart")
        table([{"k": r["k"], "d": r["d"], "d/(n − 1)": V.de(r["share"], 2), "thr": r["thr"], "Ringzeit": cell_text({"mean": r["mean"], "se": r["se"], "capped": r["capped"]}), "Vollgraph (exakt)": V.fmt_time(r["complete"]),
                "Vorteil (Faktor)": V.de(r["ratio"], 1) if r["ratio"] == r["ratio"] else "–", "abgebrochen": f"{r['capped']} von {r['total']}"} for r in rows3])
    v3 = V.verdict_e3(rows3, n3, eps3)
    show(v3)
    verdicts.append(v3)

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Die Treppe der Schwelle")
    st.caption(f"Ring mit n = {GRID.E4_N}, ε = {V.de(GRID.E4_EPS, 1)}, k = {GRID.E4_KS[0]} bis {GRID.E4_KS[-1]}, Schwelle q von {V.de(GRID.E4_QS[0], 2)} bis {V.de(GRID.E4_QS[-1], 2)}. Die Zahl der nötigen Partner thr = ⌈q·2k⌉ springt, die Zeit mit ihr.")
    rows4 = X.e4_rows(cells)
    if any(r["mean"] is not None for r in rows4):
        st.plotly_chart(Z.build_staircase(rows4), width="stretch", key="e4_chart")
        table([{"k": r["k"], "q": V.de(r["q"], 2), "thr": r["thr"], "Spielraum s = k − thr": r["spread"], "Zeit": cell_text({"mean": r["mean"], "se": r["se"], "capped": r["capped"]})} for r in rows4])
    v4 = V.verdict_e4(rows4)
    show(v4)
    verdicts.append(v4)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Gitter")
    st.caption(f"Torus mit L × L Betrieben, 4 oder 8 Nachbarn, ε = {V.de(GRID.E5_EPS[0], 1)} und {V.de(GRID.E5_EPS[1], 1)} ({len(GRID.SEEDS)} Ströme zu je {GRID.E5_RUNS} Läufen, vorgerechnet). Die Randsicht ist die Zahl der A-Partner am geraden Rand eines A-Blocks.")
    e5_eps = st.selectbox("Fehlerrate ε", GRID.E5_EPS, key="e5_eps", format_func=lambda e: V.de(e, 1))
    rows5 = X.e5_rows(cells)
    if any(r["mean"] is not None for r in rows5):
        st.plotly_chart(Z.build_grid_times(rows5, e5_eps), width="stretch", key="e5_chart")
        table([{"Nachbarn": "4" if r["kind"] == "vn" else "8", "q": V.de(r["q"], 2), "thr": r["thr"], "Randsicht": r["edge"], "ε": V.de(r["eps"], 1), "L": r["side"], "n": r["side"] ** 2, "Zeit": cell_text(r),
                "abgebrochen": "–" if r["capped"] is None else f"{r['capped']} von {r['total']}"} for r in rows5])
    v5 = V.verdict_e5(rows5)
    show(v5)
    verdicts.append(v5)

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🔬 Experiment E6: Rückwechsel")
    rows6, pair6 = _e6(settings.q)
    st.caption(f"Vollgraph, exakt, Schwelle q = {V.de(settings.q, 2)} aus der Seitenleiste. Hin: alle B → mindestens die Hälfte A. Zurück: alle A → thr_B + 2 Betriebe mit B, wobei thr_B die Zahl der B-Partner ist, "
               "ab der B beste Antwort wird (die Hälfte reicht nicht: darunter fließt das System zurück).")
    e6_eps = st.selectbox("Fehlerrate ε", C.BACK_EPS, index=1, key="e6_eps", format_func=lambda e: V.de(e, 2))
    st.plotly_chart(Z.build_back(rows6, e6_eps), width="stretch", key="e6_chart")
    table([{"ε": V.de(r["eps"], 2), "n": r["n"], "Hin": V.fmt_log10(r["hin"]), "Zurück": V.fmt_log10(r["back"]), "Verhältnis der Exponenten": V.de(r["ratio"], 2),
            "Steigung hin": "–" if r["slope_hin"] is None else V.de(r["slope_hin"], 4), "Steigung zurück": "–" if r["slope_back"] is None else V.de(r["slope_back"], 4)} for r in rows6])
    v6 = V.verdict_e6(rows6, pair6, settings.q)
    show(v6)
    verdicts.append(v6)

# --- Tab 7 ---------------------------------------------------------------------------------------------------------------------------------
with tab7:
    st.subheader("🔬 Experiment E7: Warum im Logarithmus?")
    st.caption(f"Dieselbe Zeit im Vollgraph (q = {V.de(C.LINEAR_Q, 1)}, ε = {V.de(C.LINEAR_EPS, 1)}) auf zwei Wegen: im Logarithmus und als lineares System (dichtes numpy). Das lineare System ist im Programm auf n ≤ {E.LINEAR_SAFE_MAX_N} beschränkt; hier wird es absichtlich darüber aufgerufen.")
    rows7 = _e7()
    st.plotly_chart(Z.build_linear_vs_log(rows7), width="stretch", key="e7_chart")
    table([{"n": r["n"], "Logarithmus": V.fmt_time(r["log_time"]), "lineares System": ("negativ: " if r["negative"] else "") + V.fmt_time(abs(r["linear"])), "relative Abweichung": f"{r['deviation']:.1e}".replace(".", ",")} for r in rows7])
    v7 = V.verdict_e7(rows7)
    show(v7)
    verdicts.append(v7)

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`koordinationsspiel-demo`** (Stück 1 der Reihe): die Gleichgewichte und die Schwelle q selbst, Risiko- gegen Auszahlungsdominanz.
- **`gleichgewichtsauswahl-demo`** (Stück 2): welche Konvention auf Dauer gewählt wird (stationäre Verteilung, Logit); hier geht es um die Zeit dorthin.
- **`komplexe-ansteckung-demo`**: dieselbe Schwellenregel auf Netzen, aber ohne Fehler (Runden bis zur Sättigung); hier warten wir auf Fehler.
- **`schelling-demo`**: Standortwahl mit Schwellenvorliebe; hier ist das Netz fest.
- **`rauschen-kooperation-demo`**: Fehlerrate und exakte Markov-Kette, aber im wiederholten Gefangenendilemma zwischen zwei Partnern.
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Lernregel und Eigennutz sind Modellannahmen** (beste Antwort mit Fehlerrate ε) | Gemessen ist diese Regel, nicht das Verhalten von Menschen oder Terminals. Mit Logit-Wahl statt gleichmäßiger Fehler wählen 2×2-Spiele nach `gleichgewichtsauswahl-demo` meist dieselbe Seite (60 von 63 Zellen), die Wechselzeit ist aber dort nicht gerechnet. | `gleichgewichtsauswahl-demo` |
| **Gleiche Fehlerrate für alle, Zufallswahl nur A oder B** | Typabhängige oder gegenüber den Partnern verschiedene Fehlerraten ändern die Auswahl (Stück 2) und damit die Zeit; heterogene Betriebe sind nicht gerechnet. | – |
| **Ein Betrieb je Schritt, kein gleichzeitiger Wechsel** | Bei gleichzeitiger Aktualisierung vieler Betriebe kann ein Block auf einmal kippen; gemessen ist nur die asynchrone Aktualisierung. | – |
| **q < 1/2 und feste Schwelle für alle** | Bei q ≥ 1/2 ist A nicht risikodominant; die Demo rechnet nur q von 0,10 bis 0,45. | `koordinationsspiel-demo` |
| **Das Netz ist fest, regulär und bekannt** | Wechselnde Partner, verschieden viele Partner (Nabenknoten) und Lernkosten sind nicht gerechnet; Gitter und Ring sind die Grenzfälle. | `komplexe-ansteckung-demo`, `schelling-demo` |
| **Zeit nur als Erwartungswert** | Die Wartezeit streut; ein Mittel ohne Streuung ist keine Prognose. Mittel aus Läufen, die an der Schrittgrenze abbrachen, sind verzerrt und werden nie gebildet. | – |
| **Ring und Gitter nur als Simulation** | Exakt (über alle 2^n Belegungen) sind nur Netze bis n = 12; darüber stützen sich die Zahlen auf Läufe mit festen Zufalls-Strömen. Gitter mit 8 Nachbarn und Schwelle 4 sind nicht gesättigt (Läufe brechen ab). | – |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Betriebe auf einem Netz, in dem jeder $d$ Partner hat, eine Schwelle $q$ und eine Fehlerrate $\varepsilon$; gesucht ist die erwartete Zahl der Aktualisierungen je Betrieb, bis mindestens die Hälfte das neue Format führt.

| Symbol | Bedeutung |
|---|---|
| $n,\ d$ | Zahl der Betriebe, Zahl der Partner je Betrieb |
| $q,\ \mathrm{thr}$ | Schwelle und nötige Zahl der A-Partner, $\mathrm{thr} = \lceil q\,d\rceil$ |
| $\varepsilon$ | Fehlerrate (Zufallswahl, A oder B je zur Hälfte) |
| $k$ | Zahl der Betriebe mit A (Vollgraph) |
| $r^{+}_k,\ r^{-}_k$ | Aufwärts- und Abwärtsrate je Einzelaktualisierung |
| $T$ | erwartete Zahl der Aktualisierungen je Betrieb |

**Aufwärts- und Abwärtsrate im Vollgraph** ($[\cdot]$ ist 1, wenn die Bedingung gilt, sonst 0):

$$r^{+}_k = \frac{n-k}{n}\left((1-\varepsilon)\,[k \ge \mathrm{thr}] + \frac{\varepsilon}{2}\right), \qquad r^{-}_k = \frac{k}{n}\left((1-\varepsilon)\,[k-1 < \mathrm{thr}] + \frac{\varepsilon}{2}\right)$$

**Erwartete Zeit bis zur Hälfte** $m = \lceil n/2\rceil$ von $k = 0$ aus (stationäre Gewichte $\pi_j \propto \prod r^{+}/r^{-}$, im Logarithmus aufsummiert):

$$T = \frac{1}{n}\sum_{k=0}^{m-1} \frac{1}{\pi_k\, r^{+}_k}\sum_{j=0}^{k} \pi_j$$

**Gewichte unterhalb der Schwelle.** Für $k < \mathrm{thr}$ gilt $\pi_{k+1}/\pi_k = \frac{n-k}{k+1}\cdot\frac{\varepsilon/2}{1-\varepsilon/2}$, das Verhältnis der Binomialverteilung:

$$\pi_k \propto \binom{n}{k}\left(\frac{\varepsilon}{2}\right)^{k}\left(1-\frac{\varepsilon}{2}\right)^{n-k}, \qquad k \le \mathrm{thr}$$

**Rate** (Chernoff, $n \to \infty$ mit $\mathrm{thr}/n \to q$):

$$\frac{\log_{10} T}{n} \longrightarrow \frac{\mathrm{KL}(q \,\Vert\, \varepsilon/2)}{\ln 10}$$

**Rückwechsel** (Rollen vertauscht): B ist beste Antwort genau dann, wenn weniger als $\mathrm{thr}$ Partner A führen, also mindestens $\mathrm{thr}_B = d - \mathrm{thr} + 1$ Partner B führen; Start „alle A“, Ziel $\mathrm{thr}_B + 2$ Betriebe mit B.

**Beliebiges Netz** (Zustände $s \in \{0,1\}^n$, $c_i(s)$ = Zahl der A-Partner von $i$): erwartete Schritte $t(s)$ lösen $t(s) = 1 + \frac{1}{n}\sum_i \left(p_i(s)\,t(s \cup i) + (1-p_i(s))\,t(s \setminus i)\right)$ mit

$$p_i(s) = (1-\varepsilon)\,[c_i(s) \ge \mathrm{thr}] + \frac{\varepsilon}{2}$$

und $t = 0$ ab $\lceil n/2\rceil$ Betrieben mit A; $T = t(\text{alle B})/n$.

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung, durch Aufzählung aller $2^n$ Belegungen und ein lineares System bestätigt.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from kvw_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="konventionswechsel.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
