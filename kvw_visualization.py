"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Zeiten stehen als Zehnerlogarithmus (Aktualisierungen je Betrieb): 10^77 lässt sich nicht als gewöhnliche Zahl zeichnen."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import kvw_constants as C

RING_COLORS = {1: "#2F6B65", 2: "#4c78a8", 3: "#C0762B", 4: "#8e6c8a"}
GRID_COLORS = {"vn": "#4c78a8", "moore": "#C0762B"}
STATE_COLORSCALE = [[0.0, C.COLOR_B], [1.0, C.COLOR_A]]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70 if legend else 10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


# ---------------------------------------------------------------------------------------------------------------------------- Hauptansicht
def build_trace(series: list[int], n: int, target: int, reached: bool, y_label: str = "Anteil der Betriebe mit A") -> go.Figure:
    """Verlauf eines einzelnen Laufs: Anteil der Betriebe mit der neuen Konvention über die Zeit (Aktualisierungen je Betrieb); gestrichelt das Ziel."""
    steps = np.arange(1, len(series) + 1)
    stride = max(1, len(series) // 2000)
    fig = go.Figure(go.Scatter(x=(steps / n)[::stride], y=(np.array(series) / n)[::stride], mode="lines", line=dict(color=C.COLOR_A, width=2), name="Anteil", hovertemplate="Zeit %{x:.2f}: Anteil %{y:.3f}<extra></extra>"))
    fig.add_hline(y=target / n, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="Ziel", annotation_position="top left")
    fig.update_xaxes(title_text="Aktualisierungen je Betrieb")
    fig.update_yaxes(title_text=y_label, range=[0, 1.02])
    if not reached:
        fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper", text="Ziel in diesem Lauf nicht erreicht", showarrow=False, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.88)")
    return _base(fig, legend=False)


def build_ring(state: list[int]) -> go.Figure:
    """Der Ring als Kreis: jeder Betrieb ein Punkt, grün = neues Format A, orange = altes Format B."""
    n = len(state)
    angle = 2 * math.pi * np.arange(n) / n
    fig = go.Figure(go.Scatter(x=np.cos(angle), y=np.sin(angle), mode="markers", marker=dict(size=max(6, min(22, 420 // n)), color=[C.COLOR_A if s else C.COLOR_B for s in state], line=dict(width=0.5, color="white")),
                               hovertemplate="Betrieb %{customdata}<extra></extra>", customdata=list(range(n))))
    fig.update_xaxes(visible=False, range=[-1.15, 1.15])
    fig.update_yaxes(visible=False, range=[-1.15, 1.15], scaleanchor="x", scaleratio=1)
    return _base(fig, height=360, legend=False)


def build_grid(state: list[int], side: int) -> go.Figure:
    """Das Gitter als Raster: grün = A, orange = B."""
    z = np.array(state, dtype=float).reshape(side, side)
    fig = go.Figure(go.Heatmap(z=z, colorscale=STATE_COLORSCALE, zmin=0, zmax=1, showscale=False, xgap=1, ygap=1, hovertemplate="Zeile %{y}, Spalte %{x}: %{z:.0f}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1)
    return _base(fig, height=360, legend=False)


def build_count(series_a: int, n: int) -> go.Figure:
    """Der Vollgraph als Zähler: ein Balken, wie viele der n Betriebe A führen."""
    fig = go.Figure(go.Bar(x=["A", "B"], y=[series_a, n - series_a], marker=dict(color=[C.COLOR_A, C.COLOR_B])))
    fig.update_yaxes(title_text="Betriebe", range=[0, n])
    return _base(fig, height=300, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- E1
def build_time_over_n(rows: list[dict], eps: float) -> go.Figure:
    """E1: log10 der Wechselzeit über n. Linie: Vollgraph (exakt); Punkte: Ring k = 1 bis 4 (Simulation, Mittel mit Standardfehler; ausgefüllt nur ohne abgebrochene Läufe)."""
    ns = [r["n"] for r in rows]
    fig = go.Figure(go.Scatter(x=ns, y=[r["complete_log10"] for r in rows], mode="lines+markers", name="Vollgraph (exakt)", line=dict(color=C.COLOR_RED, width=3)))
    for k in (1, 2, 3, 4):
        pts = [(r["n"], r["rings"][k]) for r in rows if r["rings"].get(k) is not None and r["rings"][k]["mean"] == r["rings"][k]["mean"]]
        if not pts:
            continue
        y = [math.log10(c["mean"]) for _, c in pts]
        err = [c["se"] / (c["mean"] * math.log(10)) for _, c in pts]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=y, mode="lines+markers", name=f"Ring k = {k} (Simulation)", line=dict(color=RING_COLORS[k], width=2),
                                 error_y=dict(type="data", array=err, visible=True),
                                 marker=dict(symbol=["circle" if c["capped"] == 0 else "circle-open" for _, c in pts], size=9)))
    fig.update_xaxes(title_text="Zahl der Betriebe n", type="log")
    fig.update_yaxes(title_text="log10 der Aktualisierungen je Betrieb")
    return _base(fig, height=420)


# ---------------------------------------------------------------------------------------------------------------------------- E2
def build_rates(rows: list[dict]) -> go.Figure:
    """E2: je ε die gemessene Steigung (exakt) neben KL(q ‖ ε/2)/ln 10 und der groben Erwartung q·log10(1/ε)."""
    labels = [f"ε = {r['eps']}".replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["measured"] for r in rows], name="gemessen (exakt, n = 800 → 1 600)", marker_color=C.COLOR_A))
    fig.add_trace(go.Bar(x=labels, y=[r["chernoff"] for r in rows], name="KL(q ‖ ε/2)/ln 10", marker_color=C.COLOR_LINE))
    fig.add_trace(go.Bar(x=labels, y=[r["rough"] for r in rows], name="grobe Erwartung q·log10(1/ε)", marker_color=C.COLOR_GREY))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Zuwachs von log10 T je Betrieb")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E3
def build_neighbourhood(rows: list[dict], n: int, eps: float) -> go.Figure:
    """E3: Vorteil des Netzes (Zeit Vollgraph / Zeit Ring) über den Anteil der gesehenen Betriebe d/(n − 1), logarithmisch; Linie bei 1 = kein Vorteil."""
    pts = [r for r in rows if r["ratio"] == r["ratio"]]
    fig = go.Figure(go.Scatter(x=[r["share"] for r in pts], y=[r["ratio"] for r in pts], mode="lines+markers", name=f"n = {n}, ε = {str(eps).replace('.', ',')}",
                               line=dict(color=C.COLOR_A, width=2), marker=dict(symbol=["circle" if r["capped"] == 0 else "circle-open" for r in pts], size=9),
                               customdata=[r["k"] for r in pts], hovertemplate="k = %{customdata}, sieht %{x:.2f}: Faktor %{y:.3g}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="kein Vorteil", annotation_position="bottom right")
    fig.update_xaxes(title_text="Anteil der anderen Betriebe, die ein Betrieb sieht: d/(n − 1)", range=[0, 1.02])
    fig.update_yaxes(title_text="Vorteil des Ringes (Faktor)", type="log")
    return _base(fig, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- E4
def build_staircase(rows: list[dict]) -> go.Figure:
    """E4: log10 der Ringzeit über q, eine Linie je k; die Stufen sind die Sprünge der ganzzahligen Schwelle."""
    fig = go.Figure()
    for k in sorted({r["k"] for r in rows}):
        pts = [r for r in rows if r["k"] == k and r["mean"] is not None and r["mean"] == r["mean"]]
        fig.add_trace(go.Scatter(x=[r["q"] for r in pts], y=[math.log10(r["mean"]) for r in pts], mode="lines+markers", name=f"k = {k}", line=dict(color=RING_COLORS.get(k - 2, C.COLOR_LINE), width=2),
                                 customdata=[[r["thr"], r["spread"]] for r in pts], hovertemplate="q = %{x}: Schwelle %{customdata[0]}, Spielraum %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title_text="Schwelle q")
    fig.update_yaxes(title_text="log10 der Aktualisierungen je Betrieb")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E5
def build_grid_times(rows: list[dict], eps: float) -> go.Figure:
    """E5: log10 der Gitterzeit über die Kantenlänge L, eine Linie je (Nachbarschaft, q); offene Marker: Läufe abgebrochen."""
    fig = go.Figure()
    for kind, q in sorted({(r["kind"], r["q"]) for r in rows}):
        pts = [r for r in rows if r["kind"] == kind and r["q"] == q and abs(r["eps"] - eps) < 1e-12 and r["mean"] is not None and r["mean"] == r["mean"]]
        if not pts:
            continue
        thr = pts[0]["thr"]
        fig.add_trace(go.Scatter(x=[r["side"] for r in pts], y=[math.log10(r["mean"]) for r in pts], mode="lines+markers",
                                 name=f"{'4' if kind == 'vn' else '8'} Nachbarn, q = {str(q).replace('.', ',')} (Schwelle {thr})", line=dict(color=GRID_COLORS[kind], width=2, dash="solid" if q < 0.35 else "dash"),
                                 marker=dict(symbol=["circle" if r["capped"] == 0 else "circle-open" for r in pts], size=9)))
    fig.update_xaxes(title_text="Kantenlänge L (n = L²)", type="log", tickvals=[4, 8, 12, 24], ticktext=["4", "8", "12", "24"])
    fig.update_yaxes(title_text="log10 der Aktualisierungen je Betrieb")
    return _base(fig, height=400)


# ---------------------------------------------------------------------------------------------------------------------------- E6
def build_back(rows: list[dict], eps: float) -> go.Figure:
    """E6: log10 der Zeit für Hin- und Rückwechsel über n (exakt, Vollgraph)."""
    pts = [r for r in rows if abs(r["eps"] - eps) < 1e-12]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["n"] for r in pts], y=[r["hin"] for r in pts], mode="lines+markers", name="Hinwechsel (alle B → A)", line=dict(color=C.COLOR_A, width=3)))
    fig.add_trace(go.Scatter(x=[r["n"] for r in pts], y=[r["back"] for r in pts], mode="lines+markers", name="Rückwechsel (alle A → B)", line=dict(color=C.COLOR_B, width=3)))
    fig.update_xaxes(title_text="Zahl der Betriebe n")
    fig.update_yaxes(title_text="log10 der Aktualisierungen je Betrieb")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E7
def build_linear_vs_log(rows: list[dict]) -> go.Figure:
    """E7: relative Abweichung des linearen Systems vom Logarithmus-Weg über n, logarithmisch; rot: das lineare System liefert eine negative Zeit."""
    fig = go.Figure(go.Bar(x=[str(r["n"]) for r in rows], y=[max(r["deviation"], 1e-16) for r in rows], marker=dict(color=[C.COLOR_RED if r["negative"] else C.COLOR_LINE for r in rows]),
                           hovertemplate="n = %{x}: Abweichung %{y:.2e}<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Betriebe n")
    fig.update_yaxes(title_text="relative Abweichung (logarithmisch)", type="log", range=[-16, 1])
    return _base(fig, legend=False)
