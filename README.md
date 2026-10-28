# Wie lange dauert der Wechsel einer Konvention? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-konventionswechsel-demo.streamlit.app/)** -->

Drittes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **Palettenformat.** Betriebe führen das neue Format A oder das alte B; A lohnt sich für einen Betrieb, sobald mindestens der Anteil **q** seiner Partner es führt (q < 1/2: A ist risikodominant). Jeder Betrieb prüft zu zufälligen Zeitpunkten und wählt die beste Antwort, mit Wahrscheinlichkeit **ε** stattdessen zufällig (A oder B je zur Hälfte). Die Demo rechnet **exakt**, wie lange es dauert, bis A bei „alle führen B“ als Start mindestens die Hälfte der Betriebe erreicht, **woher die Rate der exponentiellen Zunahme mit der Zahl der Betriebe kommt**, wo ein **Netz mit wenigen Partnern** (Ring, Gitter) die Zeit um viele Zehnerpotenzen verkürzt, und wie lange der **Rückwechsel** dauert.

Abgrenzung: `koordinationsspiel-demo` (erstes Stück) zeigt die Gleichgewichte und die Schwelle q, `gleichgewichtsauswahl-demo` (zweites Stück), welche Konvention auf Dauer gewählt wird; hier geht es um die **Zeit** dorthin. `komplexe-ansteckung-demo` rechnet dieselbe Schwellenregel auf Netzen, aber ohne Fehler (Runden bis zur Sättigung); `schelling-demo` die Standortwahl mit Schwellenvorliebe; `rauschen-kooperation-demo` eine Fehlerrate und exakte Markov-Kette im wiederholten Gefangenendilemma zwischen zwei Partnern; `replikator-demo` Populationsdrift ohne Netze. **Eigennutz und die Lernregel sind Modellannahmen**, keine Aussagen über Menschen oder Terminals; jede Zeit ist ein Erwartungswert mit Streuung, keine Prognose.

## Kernfrage

*Wie lange braucht es, bis eine bessere (risikodominante) Konvention eine schlechtere verdrängt, wenn nur Fehler sie anstoßen, und warum wächst die Zeit bei „jeder trifft jeden“ um Zehnerpotenzen mit der Zahl der Betriebe, in einem Nachbarschaftsnetz mit wenigen Partnern aber nicht?* Auswahl durch Rauschen und die Rolle der Netze sind Stoff der Literatur (Quellen unten). Eigenes dieser Demo sind (1) die **Wechselzeit als Erwartungswert, exakt gerechnet** (Warten auf Fehler, nicht Runden einer deterministischen Regel), (2) die **Rate** der Zunahme mit n im Vollgraph, **erklärt und gemessen**, (3) Ring, Gitter und Nachbarschaftsgröße gegen den Vollgraph bei **gleichem Spiel**, (4) der **Rückwechsel**, (5) das Rechenlehrstück „lineares System bricht, Logarithmus trägt“.

Zur Einordnung (Annahme: jeder Betrieb prüft einmal pro Woche): im Vollgraph mit n = 20, q = 0,4 und ε = 0,1 sind es 89.448 Aktualisierungen je Betrieb, rund 1.720 Jahre; im Ring mit zwei Partnern je Seite sind es 12,0 bis 12,7 Aktualisierungen, rund 12 Wochen (n = 40 bis 320).

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gemessen (`tests/test_claims.py`). Die Ringe und Gitter sind Simulationen mit vier festen Zufalls-Strömen (SplitMix64, `data/measurements.json` mit Erzeuger-Hash); der Vollgraph, der Rückwechsel und das Lehrstück sind exakt.

| Nr. | Hypothese | Antwort | Beleg |
|---|---|---|---|
| H1 | Vollgraph: Wechselzeit wächst exponentiell mit n | **Ja** | ε = 0,1, q = 0,4, n = 10 / 20 / 40 / 80 / 160 / 320: 496 / 89.448 / 4,2·10^9 / 1,3·10^19 / 2,0·10^38 / 6,1·10^76 Aktualisierungen je Betrieb (exakt im Logarithmus); bei n = 1 600 sind es 10^385,5. Gegen Handrechnung (n = 3: 5,5556 und 10,5263), exakte Brüche und die Aufzählung aller 2ⁿ Belegungen geprüft (`test_h1_*`) |
| H2 | Ring mit k = 1 und 2 Partnern je Seite ist ab etwa n = 40 praktisch unabhängig von n | **Ja, mit Einschränkung bei kleinem ε** | Spanne (größter gegen kleinsten Mittelwert) bei k = 1 ab n = 20: 4,4 / 2,6 / 1,7 % für ε = 0,2 / 0,1 / 0,05; bei k = 2 ab n = 40: 1,6 / 6,2 / 12,6 %. Bei ε = 0,05 fällt k = 2 von 25,5 (n = 40) auf 22,6 (n = 320) noch (`test_h2_*`) |
| H2b | Ring mit k ≥ 3 ist bei kleinem ε „unabhängig von n“ | **Nein** | ε = 0,1: k = 3 fällt von 43,1 (n = 40) auf 32,0 (n = 320), k = 4 von 254,7 auf 89,5, ε = 0,05 und k = 4 von 3.064 auf 454. Richtig ist „nicht wachsend in n“. Bei n = 10 ist der Ring mit k = 4 (8 von 9 Partnern) langsamer als der Vollgraph: 63,5 gegen 50,2 (ε = 0,2), 677,3 gegen 495,9 (ε = 0,1) (`test_h2b_*`) |
| H3 | Die Rate je Betrieb ist KL(q ‖ ε/2)/ln 10 (Binomialverteilung der Fehler unterhalb der Schwelle) | **Ja** (eigene Herleitung, nicht gegen die Literatur geprüft) | Steigung von log₁₀ T je Betrieb zwischen n = 800 und 1 600 für ε = 0,2 / 0,1 / 0,05 / 0,02: gemessen 0,1350 / 0,2413 / 0,3549 / 0,5101, Formel 0,1352 / 0,2415 / 0,3551 / 0,5103, in allen vier um 0,0002 darunter. Für q = 0,1 / 0,2 / 0,3 / 0,4 / 0,45 (ε = 0,1, n = 400 → 800): 0,0086 / 0,0603 / 0,1402 / 0,2411 / 0,2985 gegen 0,0090 / 0,0607 / 0,1406 / 0,2415 / 0,2989 (`test_h3_*`) |
| H3b | Die Rate ist q·log₁₀(1/ε) | **Nein** | Die grobe Erwartung ist das 2,07- / 1,66- / 1,47- / 1,33-fache der gemessenen Rate (ε = 0,2 / 0,1 / 0,05 / 0,02) (`test_h3b_*`) |
| H4 | T ∝ ε^(−a) mit a ≈ thr (nötige Zahl der A-Partner) | **Ja, mit Abschlag** | n = 10 / 20 / 40 / 80 (thr 4 / 8 / 16 / 32): a = 3,67 / 7,45 / 15,01 / 30,11, also das 0,92- bis 0,94-fache von thr (`test_h4_*`) |
| H5 | Das lineare System bricht ab etwa n = 50, wird bei n = 70 negativ | **Ja, mit anderen Zahlen** | Relative Abweichung gegen den Logarithmus (ε = 0,1, q = 0,4, numpy-Löser, Windows): 3,0·10⁻⁹ (n = 30) / 5,3·10⁻⁷ (40) / 1,0·10⁻⁴ (50) / 1,9·10⁻² (60); bei n = 70 −7,0·10^15 statt 5,6·10^16. Wo es bricht, hängt an der Routine und Plattform; die Tests prüfen nur Bänder (`test_h5_*`) |
| H6 | Der Vorteil des Netzes hängt am Anteil der gesehenen Betriebe d/(n − 1), nicht an der Ringform | **Teilweise** | Erst bei fast voller Sicht endet er: höchstens der Faktor 2 bei Anteil 0,84 (n = 20, ε = 0,2: 1,78) bzw. 0,90 (n = 30: 1,94); der Ring ist langsamer als der Vollgraph bei 0,95 (0,74) bzw. 0,97 (0,66). Bei etwa halber Sicht noch 42,3 (n = 20, k = 5) und 43,7 (n = 30, k = 8). Der Anteil allein bestimmt den Faktor nicht: bei Anteil 0,41 ist er 421 (n = 30, k = 6) und 1.039 (n = 40, k = 8) (`test_h6_*`) |
| H7 | Die Zeit hängt nur an thr = ⌈q·d⌉ und k, fällt mit dem Spielraum s = k − thr und wächst bei Spielraum 0 je Schwellenstufe um 2,7 bis 3,5 | **Ja, mit anderen Faktoren** | Zellen mit gleichem k und thr sind zahlengleich. thr = 4: s = 0 / 1 / 2: 36,5 / 15,8 / 9,2. Bei Spielraum 0 (thr = k = 3 bis 6): 14,2 / 36,5 / 116,9 / 362,4, Faktoren 2,6 / 3,2 / 3,1 (Ring n = 30, ε = 0,2) (`test_h7_*`) |
| H8 | Gitter: unabhängig von L ab L = 12 bei Schwelle 1 und 2 (4 Nachbarn) und 3 (8 Nachbarn) | **Ja** | Von L = 12 auf 24 ändert sich die Zeit in allen 6 Zellen (zwei ε) um höchstens 6,6 %; L = 4 ist deutlich langsamer (4 Nachbarn, Schwelle 2, ε = 0,2: 10,1 gegen 7,7) (`test_h8_*`) |
| H8b | Gitter mit 8 Nachbarn und Schwelle 4 | **Offen, nicht gesättigt** | ε = 0,1: 1.706 / 12.126\* / 5.162\* / 1.228\* für L = 4 / 8 / 12 / 24 (\*: 1 / 1 / 3 von 240 Läufen an der Schrittgrenze abgebrochen, Mittel nach unten verzerrt); ε = 0,2: 88 / 82 / 48 / 36, fällt mit L. Für die Abhängigkeit von L unbrauchbar, keine Aussage (`test_h8b_*`) |
| H9 | Rückwechsel: Exponent rund doppelt, Rate KL(1 − q ‖ ε/2) | **Ja** | Verhältnis der Exponenten zurück/hin bei n = 640: 2,42 / 2,06 / 1,90 (ε = 0,2 / 0,1 / 0,05). Steigung zurück 0,4968 (n = 320 → 640, ε = 0,1) gegen KL(0,6 ‖ 0,05)/ln 10 = 0,4972 (hin: 0,2410 gegen 0,2415) (`test_h9_*`) |
| H10 | Keimbildung im Ring: T ∝ ε^(−thr/2) | **Nur als Richtung** | Exponent zwischen ε = 0,1 und 0,05 bei n = 320 für thr = 1 / 2 / 3 / 4: 0,49 / 0,86 / 1,44 / 2,34 gegen 0,5 / 1 / 1,5 / 2; bei thr = 4 (k = 4) ist n = 320 nicht gesättigt (die Zeit fällt von n = 160 auf 320 noch von 103,1 auf 89,5) (`test_h10_*`) |

### Korrekturen gegenüber dem Plan

- **Rate:** Die grobe Erwartung q·log₁₀(1/ε) aus der ersten Vormessung ist falsch (H3b); die Rate ist KL(q ‖ ε/2)/ln 10. Der Wert 0,24 bei n = 60 bis 120 war ein Näherungswert auf dem Weg zu 0,2413 bei n = 800 bis 1 600.
- **Ring „unabhängig von n“** gilt nur für k = 1, 2 und nicht bei jedem ε (H2, H2b).
- **Lineares System:** Der Plan nannte 3·10⁻³ bei n = 50, 0,175 bei n = 60 und −1,8·10^15 bei n = 70; mit dem Löser dieser Demo sind es 1,0·10⁻⁴, 1,9·10⁻² und −7,0·10^15. Die Schwelle hängt an der Lösungsroutine (H5).
- **Vorteil des Netzes:** Der Plan sagte „Faktor ≤ 2 erst bei d ≥ 0,9·(n − 1)“; bei n = 20 gilt er schon bei 0,84 (H6). „Hängt nur am Anteil“ trifft nicht zu.
- **Ringzahlen mit 240 statt 120 Läufen:** ε = 0,1, k = 2, n = 10 bis 320: 21,5 / 14,8 / 12,0 / 12,7 / 12,5 / 12,4 gegen 21,7 / 15,9 / 12,8 / 12,6 / 12,3 / 12,0 der Vormessung (bei n = 20 liegt der Plan-Wert 8 % darüber).
- **Faktoren je Schwellenstufe:** 2,6 / 3,2 / 3,1 statt 2,7 / 3,2 / 3,5 (H7).
- **Rückwechsel-Ziel:** Hin ist „mindestens die Hälfte A“; zurück reicht die Hälfte nicht (unterhalb von thr_B = n − thr fließt das System zurück), das Ziel ist thr_B + 2 B-Betriebe. Das ist eine Setzung der Vormessung, kein Ergebnis.
- **Ports und Namen:** Port 9205, Präfix `kvw`, Linie `koordination`.
- **Quellen** aus dem Plan sind nur dort aufgenommen, wo ein Abstract geprüft wurde (siehe unten).

## Modell

Ein Betrieb mit d Partnern führt A, wenn mindestens thr = ⌈q·d⌉ Partner A führen (bei Gleichstand gewinnt A; die Toleranz 10⁻⁹ verhindert, dass ein Gleitkomma-Rest wie 0,28·25 = 7,000000000000001 die Schwelle um eins anhebt), mit Wahrscheinlichkeit ε wählt er stattdessen A oder B je zur Hälfte. Je Schritt entscheidet ein gleichverteilt gezogener Betrieb neu (asynchron). Start: alle B; Ziel: mindestens ⌈n/2⌉ Betriebe mit A; Zeit: Schritte geteilt durch n. Im Vollgraph (jeder trifft jeden) zählt nur die Zahl k der A-Betriebe, eine Geburts-Sterbe-Kette mit den Raten r⁺ₖ = (n − k)/n·((1 − ε)·[k ≥ thr] + ε/2) und r⁻ₖ = k/n·((1 − ε)·[k − 1 < thr] + ε/2); die erwartete Zeit steht im 📐-Expander. Unterhalb der Schwelle ist das Verhältnis der Gewichte das der Binomialverteilung Bin(n, ε/2) – daraus folgt die Rate (H3). **Spielraum s** = Randsicht − thr: Randsicht ist die Zahl der A-Partner am geraden Rand eines A-Blocks (Ring: k; Gitter mit 4 Nachbarn: 1, mit 8: 3). Bei s ≥ 0 rückt die Kante ohne Fehler vor; die Ecken eines Blocks prüft der Spielraum nicht (bei 8 Nachbarn und Schwelle 4 erodieren sie, H8b).

## Methodik

- **Exakt:** Vollgraph im Logarithmus (O(n), n = 20 000 in rund einer Millisekunde; einmal gemessen, nicht getestet); lineares System (nur n ≤ 40, Lehrstück bis n = 70); beliebige Netze mit n ≤ 12 über alle 2ⁿ Belegungen (dichtes numpy, rund eine Sekunde).
- **Simulation:** alle Läufe zugleich (numpy über die Läufe), aller Zufall aus SplitMix64 (`kvw_rng.py`, wörtliche Kopie aus `ab-test-demo`), je Lauf ein eigener Strom, drei Zahlen je Schritt; Läufe an der Schrittgrenze werden gemeldet, nie gemittelt. Der Vollgraph läuft gezählt (nur k), Ring und Gitter agentenweise. Eine einfache Schleife (`simulate_one`) liefert dieselben Zahlen wie die vektorisierte Fassung und dient als Referenz und für das Bild eines Laufs.
- **Vorgerechnet** (`tools/make_measurements.py`, `data/measurements.json` mit Erzeuger-Hash über Raster, Simulation, Netze, Modell und Generator): E1, E3, E4, E5 mit vier festen Zufalls-Strömen (100001 bis 100004), je Zelle 40 oder 60 Läufe je Strom. Ändert sich der Rechencode, stimmt der Hash nicht mehr und die App zeigt die Zellen nicht.
- **Experimente:** E1 Zeit über n (Vollgraph exakt, Ring simuliert), E2 Welche Rate?, E3 Wie groß darf die Nachbarschaft sein?, E4 Die Treppe der Schwelle, E5 Gitter, E6 Rückwechsel, E7 Warum im Logarithmus.

## Ehrliche Grenzen

- **Alle Aussagen gelten für diese Lernregel** (beste Antwort mit gleichmäßiger Fehlerrate ε, nur A oder B als Zufallswahl) in einem festen, regulären Netz mit asynchroner Aktualisierung, nicht für Menschen oder Terminals. Die Auswahl der Konvention (welche gewinnt, Stück 2) ist hier vorausgesetzt: q < 1/2, A risikodominant. In `gleichgewichtsauswahl-demo` wählen gleichmäßige Fehler im 2×2-Spiel in 60 von 63 Zellen dieselbe Seite wie die Logit-Wahl (H8 dort); die Wechselzeit mit Logit-Wahl ist hier nicht gerechnet.
- **Zeiten sind Erwartungswerte.** Die Wartezeit streut: im Ring mit n = 80, k = 2, ε = 0,1 liegen das 10-%-Quantil bei 8,0, der Median bei 12,0 und das 90-%-Quantil bei 18,3 gegen ein Mittel von 12,7 (240 Läufe); bei n = 20 sind es 5,0 / 12,0 / 29,5 gegen 14,8. Ein Mittel ohne Streuung ist keine Prognose. Eine Zahl wie „1.720 Jahre“ folgt aus der Annahme einer Prüfung je Betrieb und Woche und ist ein Größenmaß, kein Befund über Märkte.
- **Ring, Gitter: Simulation.** Exakt sind nur Netze bis n = 12; darüber stützen sich die Zahlen auf Läufe mit vier festen Strömen (240 oder 160 Läufe je Zelle). Zellen mit abgebrochenen Läufen (offene Marker, Sternchen) sind nach unten verzerrt. **Nicht gesättigt:** Gitter mit 8 Nachbarn und Schwelle 4 (H8b), Ring mit k = 4 und kleinem ε (H10).
- **Nachbarschaft:** der Spielraum prüft die gerade Kante, nicht die Ecken und nicht, wie viele Fehler eine Kante braucht (4 Nachbarn, Schwelle 2 hat s = −1 und ist trotzdem schnell, H8).
- Das **lineare System** (E7) ist in der Programmfunktion auf n ≤ 40 beschränkt; die Tabelle ruft es absichtlich darüber auf. Wo es bricht, hängt an Routine und Plattform.
- Die **Rate** ist für die Hälfte als Ziel und die Chernoff-Näherung hergeleitet; die Herleitung ist eigene Rechnung, durch Zahlen bestätigt (auf 0,0003 für q = 0,4 und vier ε), nicht gegen die Literatur geprüft.
- Das **Rückwechsel-Ziel** thr_B + 2 ist gesetzt; mit einem anderen Ziel verschieben sich die Zeiten, die Steigung nicht.
- Mausklicks auf die Diagramme sind nicht getestet (die Diagramme haben keine Auswahl); der Schieberegler für das Bild eines Laufs ist im AppTest, nicht mit echtem Mausklick geprüft.
- Nur an erfundenen Parametern geprüft; keine Messung an Terminals oder Menschen.

## Quellen

Am Abstract geprüft (nicht im Volltext gelesen): Kandori, Mailath, Rob (1993), „Learning, Mutation, and Long Run Equilibria in Games“, *Econometrica* 61(1), 29–56 (Abstract: endliche Population mit ständigen Mutationen; die Menge der Gleichgewichte schrumpft auf „langfristige Gleichgewichte“, in 2×2-Spielen das risikodominante). Ellison (1993), „Learning, Local Interaction, and Coordination“, *Econometrica* 61(5), 1047–1071 (Abstract: bei gleichmäßiger Paarung in großen Populationen bestimmt die Vorgeschichte das Ergebnis, bei lokaler Interaktion mit wenigen Nachbarn können die Kräfte der Evolution es bestimmen; für endliche Zeithorizonte zählt die Konvergenzgeschwindigkeit). Ellison (2000), „Basins of Attraction, Long-Run Stochastic Stability, and the Speed of Step-by-Step Evolution“, *Review of Economic Studies* 67(1), 17–45 (Abstract: Radius und modifizierter Coradius begrenzen die Geschwindigkeit; Evolution ist schneller über eine Folge kleiner Schritte zwischen Zwischenzuständen). Die Zahlen und die Herleitung der Rate in dieser Demo sind eigene Rechnung und stammen nicht aus diesen Arbeiten. Nicht geprüft und nicht verwendet: Young (1993), Blume (1993).

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `test_exact.py`, `test_sim.py`, `test_graphs.py`, `test_model.py`): n = 3 im Vollgraph (Zeit je Betrieb 1/ε + 1/(2(1 − ε/2)) = 5,5556 bei ε = 0,2 und 10,5263 bei ε = 0,1), Ring n = 4 mit k = 1 (4,2105), Raten von Hand, Schwellen mit Gleitkomma-Resten, Spiegelung, Randsicht an einer echten Kante.
- **Orakel** (`tests/test_oracle_konventionswechsel.py`, `tests/test_oracle_verlauf.py`): Logarithmus-Weg gegen das lineare System (n bis 40), gegen eine eigene Aufzählung aller Belegungen mit Schleifen und scipy.sparse (n bis 12; Vollgraph, Ring, Torus 3 × 4) und gegen exakte Bruchrechnung (Gauß-Elimination mit `fractions`); der Rückwechsel gegen die ungespiegelte Dynamik im 2ⁿ-Raum; exakte Zeit gegen die Simulation (3 000 Läufe je Zelle, feste Zufalls-Seeds, Marge 4 Standardfehler); Steigung gegen die Chernoff-Rate; Binomial-Eigenschaft der Gewichte; Verlauf eines Laufs, Zusammenfassung der Läufe und Messzellen gegen wörtliche Schleifen und `statistics`.
- **Referenz der Simulation:** die vektorisierte Fassung gleicht Lauf für Lauf einer wörtlichen Schleife mit dem skalaren SplitMix64-Strom (auch über die Blockgrenze hinweg und unabhängig von der Zahl der gleichzeitigen Läufe); die gezählte Fassung im Vollgraph gleicht einer Schleife über k.
- **Randfälle** (`test_edge_cases.py`, `test_gaps.py`): n = 2, kleinste und größte Reglerwerte in allen vier Netzen, Ring mit n = 2k + 1 und Gitter 3 × 3 mit 8 Nachbarn (= Vollgraph), Ziel bei ungeradem n, Rückwechsel mit Ziel n, alle Läufe abgebrochen, Gleichstände an den Grenzen der Verdikte, Ziel 1 und Ziel n, Fehlermeldungen mit Text.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README. **Verdikte** (`test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten. **Messreihen** (`test_measurements.py`, `test_experiments.py`): Hash, Raster, drei Zellen exakt neu gerechnet, Zweig-Tests (jede Spalte weicht ab).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich je Netz und Richtung, beide ▶️-Knöpfe einzeln), Permalink, PDF, Presets samt Hilfetext-Aussagen, Layout, Footer, Laufzeit-Importe.
- **Wie die Zahlen gesichert sind:** die exakte Zeit im Vollgraph bis n = 12 auf drei voneinander unabhängigen Rechenwegen (Logarithmus, lineares System, Aufzählung aller Belegungen), bis n = 40 auf zwei; darüber nur der Logarithmus-Weg, dessen Steigung gegen die Chernoff-Rate und dessen kleine Fälle gegen die Handrechnung geprüft sind. Ring und Gitter: exakt nur bis n = 12, darüber Simulation gegen diese exakten Werte (Marge 4 Standardfehler) und Lauf für Lauf gegen eine wörtliche Schleife. Jede README-Zahl steht in `test_claims.py`, jede Schwelle und jede Zahl im Text der Verdikte wird an handgesetzten Werten geprüft, Randfälle und inklusive Grenzen stehen in `test_edge_cases.py` und `test_gaps.py`. Ob jede Aussage eines Tests wirklich etwas fängt, ist mit keinem eigenen Verfahren gemessen.

## Tests

403 Tests (`pytest tests`), rund 46 Sekunden lokal; die langsamsten sind die AppTest-Läufe über alle Regler und die Orakel gegen die Simulation. Zusammensetzung: Kern gegen Handrechnung 71 (`test_exact.py` 12, `test_sim.py` 15, `test_graphs.py` 20, `test_model.py` 16, `test_trace.py` 8), Orakel 118 (`test_oracle_konventionswechsel.py` 109, `test_oracle_verlauf.py` 9), Claims 25, Verdikte und Formate 37, Randfälle 38 (`test_edge_cases.py` 12, `test_gaps.py` 26), Messreihen 29 (`test_measurements.py` 10, `test_experiments.py` 19), Szenario und SplitMix64-Kopie 18, Abbildungen und PDF 16, Presets 12, Oberfläche 39 (`test_app.py` 18, `test_layout.py` 18, `test_footer.py` 2, `test_runtime_imports.py` 1).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht und sieben Tabs mit je einem Experiment |
| `kvw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kvw_graphs.py` | Netze: Vollgraph, Ring, Torus (4 oder 8 Nachbarn), Randsicht |
| `kvw_model.py` | Entscheidungsregel: Schwelle, beste Antwort, Spiegelung |
| `kvw_exact.py` | Exakte Zeiten (Logarithmus, lineares System, 2ⁿ-Raum), Chernoff-Rate |
| `kvw_sim.py` | Simulation (vektorisiert über die Läufe), einfache Referenzschleife, Bilder |
| `kvw_trace.py` | Verlauf eines Laufs für die Hauptansicht |
| `kvw_scenario.py` | Netz aus den Reglern (Rundung, Begrenzung) |
| `kvw_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo`) |
| `kvw_grid.py` | Messraster und Auswertung einer Zelle |
| `kvw_experiments.py` | Die Reihen E1 bis E7, Messdatei mit Erzeuger-Hash |
| `kvw_evaluation.py` | Einstellungen, Ansichten, Simulationsläufe, Verdikte als reine Funktionen |
| `kvw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `kvw_pdf_export.py` | PDF-Export (fpdf2) |
| `kvw_layout.py` | Fester Kopf (Stylesheet) |
| `kvw_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `data/measurements.json` | Vorgerechnete Zellen für E1, E3, E4, E5 mit Erzeuger-Hash |
| `tools/make_measurements.py` | Erzeugt die Messdatei neu (rund eine Viertelstunde) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Claims, Verdikte, Randfälle, Presets, Szenario und Kopie, Messreihen, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Ausbreitung ohne Fehler** (deterministische Schwellenregel, Runden bis zur Sättigung): siehe `komplexe-ansteckung-demo`. **Standortwahl:** siehe `schelling-demo`. **Logit-Auswahl und welche Konvention gewinnt:** siehe `gleichgewichtsauswahl-demo`.
- **Gleichzeitige Aktualisierung aller Betriebe**, **verschieden schnelle oder verschieden fehleranfällige Betriebe**, **wechselnde Partner**, **Netze mit verschieden vielen Partnern** (Nabenknoten, Zufallsnetze), **Lernkosten und Gedächtnis**.
- **Exakte Zeiten für Ring und Gitter mit n > 12** (der Zustandsraum 2ⁿ).
- **Menschenexperimente:** Eigennutz und Lernregeln sind Modellannahmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Demo läuft mit Python 3.12 oder neuer (CI: 3.12). In der Streamlit Cloud ist die Python-Version in den App-Einstellungen gewählt. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
