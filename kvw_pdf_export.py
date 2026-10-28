"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Netz, Zeiten und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import kvw_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "λ": "lambda", "σ": "sigma", "π": "pi", "Φ": "Phi", "Δ": "Delta", "×": "x", "↔": "<->", "≈": "~", "∝": "~", "‖": "||", "ℓ": "l"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    s, spec = view.settings, view.spec
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Wie lange dauert der Wechsel einer Konvention?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    way = "Rückwechsel (alle A -> B)" if view.back else "Hinwechsel (alle B -> A)"
    edge = "-" if spec.edge is None else f"{spec.edge} (Spielraum {spec.spread})"
    pdf.multi_cell(0, 5, _latin(f"Netz: {s.network}, n = {spec.n}, {spec.d} Partner je Betrieb. Schwelle q = {V.de(s.q, 2)} -> {spec.thr} A-Partner nötig. Fehlerrate eps = {V.de(s.eps, 2)}. {way}. "
                                f"Randsicht: {edge}. Zufalls-Seed {s.seed}, {s.runs} Läufe, Schrittgrenze {V.thousands(s.budget)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Wechselzeit", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if view.log10_time is not None:
        pdf.multi_cell(0, 5, _latin(f"Exakt (Vollgraph): {V.fmt_log10(view.log10_time)} Aktualisierungen je Betrieb."), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 5, _latin("Kein Vollgraph: Die Zeit steht in der Simulation (siehe Verdikte)."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Eigennutz und Lernregel (beste Antwort mit Fehlerrate) sind Modellannahmen, keine Aussagen über Menschen oder Terminals. Jede Zeit ist ein Erwartungswert mit Streuung, keine Prognose."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
