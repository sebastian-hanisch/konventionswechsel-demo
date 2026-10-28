"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): Vollgraph exakt im Logarithmus O(n) (n = 1 000 in Millisekunden, n = 20 000 in 0,02 s);
Lineares System O(n³) nur für Tabellenwerte bis n = 70 (E7); Netze mit n <= 12: dichtes numpy über die Übergangszustände (bis 1 586 Zeilen: rund 1 s);
Simulation: Läufe gleichzeitig, rund 20 000 bis 50 000 gemeinsame Schritte je Sekunde (Ring n = 1 000 mit k = 14: 28 Partner je Schritt und Lauf), Budget je Knopf höchstens 3·10^5 Schritte;
Verlauf eines einzelnen Laufs: höchstens TRACE_CAP Schritte in einer Python-Schleife (unter einer Sekunde)."""

from kvw_grid import SEEDS as SWEEP_SEEDS  # noqa: F401  (feste Zufallsströme der Messreihen, Definition im Raster)

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 4, 1000, 20       # Betriebe; Gitter: n wird auf die nächste Quadratzahl L² mit 3 <= L <= 31 gerundet
K_MIN, K_MAX, DEFAULT_K = 1, 14, 2          # Partner je Seite im Ring (d = 2k); begrenzt auf (n − 1)//2
Q_MIN, Q_MAX, DEFAULT_Q = 0.10, 0.45, 0.40  # Anteil der Partner, ab dem A besser ist (q < 1/2: A ist risikodominant)
EPS_MIN, EPS_MAX, DEFAULT_EPS = 0.02, 0.30, 0.10     # Fehlerrate: mit dieser Wahrscheinlichkeit wählt ein Betrieb zufällig
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 10, 200, 60       # Läufe der Simulation
SIM_BUDGETS = (30_000, 100_000, 300_000)             # Schrittgrenze je Lauf (alle Läufe rechnen gleichzeitig)
DEFAULT_BUDGET = 100_000
SEED_MAX, DEFAULT_SEED = 999999, 35
GRID_SIDE_MIN, GRID_SIDE_MAX = 3, 31                 # Kantenlänge des Gitters

NETWORKS = ("Vollgraph (offene Börse)", "Ring (Korridor)", "Gitter, 4 Nachbarn", "Gitter, 8 Nachbarn")
NETWORK_KIND = {NETWORKS[0]: "complete", NETWORKS[1]: "ring", NETWORKS[2]: "vn", NETWORKS[3]: "moore"}
DIRECTIONS = ("Hinwechsel: alle B → A", "Rückwechsel: alle A → B")

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------------
TRACE_CAP = 100_000             # Schritte des einzelnen Verlaufs (Bild); danach „Ziel nicht erreicht“
SNAPSHOT_COUNT = 40             # Zeitpunkte, an denen das Bild der Belegung gezeigt wird
LINEAR_NS = (10, 20, 30, 40, 50, 60, 70)           # E7
LINEAR_EPS, LINEAR_Q = 0.1, 0.4
RATE_EPS = (0.2, 0.1, 0.05, 0.02)                  # E2
RATE_N_LOW, RATE_N_HIGH = 800, 1600                # E2: Steigung zwischen diesen n
EXPONENT_NS = (10, 20, 40, 80)                     # E2: Abhängigkeit von ε bei festem n
EXPONENT_EPS = (0.1, 0.05)
BACK_NS = (20, 40, 80, 160, 320, 640)              # E6
BACK_EPS = (0.2, 0.1, 0.05)
WEEKS_PER_YEAR = 52                                 # Einordnung: eine Prüfung je Betrieb und Woche
CHECK_RUNS = 120                                    # Läufe der Formel-gegen-Simulation-Prüfung (drei Netze mit je 12 Betrieben, exakt über alle 2^12 Belegungen)
CHECK_EPS = 0.3                                     # feste Fehlerrate dieser Prüfung, damit die Läufe kurz bleiben
CHECK_CAP = 300_000                                 # Schrittgrenze je Lauf der Prüfung
CHECK_SEED = 7
SIM_STEPS_PER_SECOND = 15_000                       # gemeinsame Schritte je Sekunde im ungünstigsten Fall (8 Nachbarn, n = 961, 200 Läufe: 300 000 Schritte in rund 20 s)

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_A = "#2F6B65"          # neues Format A
COLOR_B = "#C0762B"          # altes Format B
COLOR_LINE = "#4c78a8"
COLOR_RED = "#d62728"
COLOR_GREY = "rgba(150,150,150,0.65)"

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {
    "network": NETWORKS[0], "n": DEFAULT_N, "k": DEFAULT_K, "q": DEFAULT_Q, "eps": DEFAULT_EPS, "direction": DIRECTIONS[0],
    "runs": DEFAULT_RUNS, "budget": DEFAULT_BUDGET, "seed": DEFAULT_SEED,
}
PRESETS = {
    "Offene Börse (Vollgraph, n = 20)": dict(BASE),
    "Korridor (Ring, k = 2, n = 80)": {**BASE, "network": NETWORKS[1], "n": 80, "k": 2},
    "Regionen (Gitter 12 × 12)": {**BASE, "network": NETWORKS[2], "n": 144, "q": 0.25},
    "Fast jeder sieht jeden (Ring, k = 14, n = 30)": {**BASE, "network": NETWORKS[1], "n": 30, "k": 14, "eps": 0.2},
    "Rückwechsel (Vollgraph, n = 40)": {**BASE, "n": 40, "direction": DIRECTIONS[1]},
}
PRESET_HELP = {
    "Offene Börse (Vollgraph, n = 20)": "Jeder Betrieb trifft jeden (n = 20, q = 0,4, ε = 0,1). Die Wechselzeit ist exakt gerechnet; sie liegt weit über dem Budget der Simulation.",
    "Korridor (Ring, k = 2, n = 80)": "Jeder Betrieb sieht nur die zwei Nachbarn links und rechts. Die Schwelle q = 0,4 verlangt 2 von 4 Partnern; eine gerade Kante rückt ohne Fehler vor (Spielraum 0). Die Simulation erreicht das Ziel.",
    "Regionen (Gitter 12 × 12)": "144 Betriebe auf einem Torus mit 4 Nachbarn, Schwelle q = 0,25: ein A-Nachbar genügt. Eine gerade Kante sieht einen A-Nachbarn und rückt vor.",
    "Fast jeder sieht jeden (Ring, k = 14, n = 30)": "Jeder Betrieb sieht 28 der 29 anderen (ε = 0,2): der Vorteil des Netzes ist fast aufgebraucht (Experiment E3).",
    "Rückwechsel (Vollgraph, n = 40)": "Die Gegenrichtung: von „alle A“ zurück zu B. Die Schwelle der gespiegelten Kette ist höher, der Exponent der Wechselzeit rund doppelt so groß (Experiment E6).",
}
