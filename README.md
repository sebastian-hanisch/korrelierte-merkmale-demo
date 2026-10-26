# Korrelierte Merkmale: wem gehört die Wichtigkeit? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-korrelierte-merkmale-demo.streamlit.app/)** -->

Zehntes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: eine Korrelation zwischen **Merkmalen** verändert, was „Wichtigkeit“ bedeutet), neben [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo) und [randabhaengigkeit-demo](https://github.com/sebastian-hanisch/randabhaengigkeit-demo). Es ist die **erste Demo des Portfolios mit Wichtigkeit unter Korrelation der Merkmale**: zwei **Zwillingskennzahlen A und B** eines Standorts (z. B. die Auslastung zweier Messpunkte an derselben Rampe) mit der Korrelation ρ, dazu unabhängige Rauschkennzahlen, und ein **Random Forest**, der die Wartezeit vorhersagt. Die Fragen: **Wem gehört die Wichtigkeit, wenn A und B (fast) dasselbe wissen? Warum ändert sich die Aufteilung von Datensatz zu Datensatz? Was erzeugt das Mischen eines Zwillings, und welche Gegenmittel gibt es?**

## Warum dieses Problem

Die Permutationswichtigkeit (eine Spalte mischen, den Fehler messen) gilt als verlässlicher als die Gini-Wichtigkeit, bei **unabhängigen** Merkmalen. Bei Zwillingen kippt das: **dieselbe Information, bei jedem Neuziehen der Daten eine andere Aufteilung.** Befunde: bei ρ = 0,99 und tragenden Zwillingen bleibt der mittlere Anteil von A bei 49,0 %, doch er streut mit **0,181** von Datensatz zu Datensatz, und in etwa einem Drittel der Läufe (**33,0 %** bei 100 Wäldern, Unsicherheit etwa ± 5 Prozentpunkte) gehört einem der beiden mehr als 70 %; mehr Bäume beruhigen die Streuung von Datensatz zu Datensatz kaum (0,163 bei 100 Bäumen), den Zufall des Waldes bei gleichen Daten schon. Mischt man einen Zwilling allein, entstehen **Zeilen aus einem Bereich, in dem nur etwa 1 % der echten Zeilen liegen**: bei ρ = 0,99 liegen **77,1 %** der Zeilen außerhalb des 3-Sigma-Bereichs. Und die Faustregel „korrelierte Merkmale verwässern die Wichtigkeit“ gilt bis ρ = 0,9 nur, wenn **beide** Zwillinge Information tragen: ist B nur eine verrauschte Kopie von A, bleibt B bis ρ = 0,9 bei höchstens 0,002; ab ρ = 0,99 verteilt der Wald die Wichtigkeit auch auf eine bloße Kopie.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-10-korrelierte-merkmale.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Kern, anderer Ziehungsreihenfolge und eigener Seed-Ableitung. Gebaut ist alles mit SplitMix64 und dem Wald aus `random-forest-demo`; die Zahlen weichen leicht ab, die Befunde bleiben, mit diesen Ausnahmen:

- **Streuung der Aufteilung:** 0,143 (ρ = 0,9) und 0,181 (ρ = 0,99) (Plan: 0,146 und 0,175), bei ρ = 0,999 **0,135** (Plan: 0,120); Läufe mit mehr als 70 : 30: 17 / 33 / 20 % (Plan: 20 / 23 / 7 %).
- **Nur A trägt bei:** B bekommt 0,080 bei ρ = 0,99 und 0,504 bei 0,999 (Plan: 0,068 und 0,471), A fällt auf 0,992 (Plan: 1,028).
- **Mehr Bäume:** bei gleichen Daten fällt die Streuung von 0,091 auf 0,028 (Plan: 0,082 auf 0,023); bei **neuen Daten sinkt sie nur von 0,200 auf 0,163** (Plan: konstant 0,18): mehr Bäume beruhigen sie **kaum**, ein kleiner Rest an Trend ist da.
- **Gruppierung nach Spearman:** bei ρ = 0,9 findet die Schwelle 0,9 das Paar in 17 % der Läufe (Plan: 16 %).
- **Lineares Modell:** mit angepassten Koeffizienten streut die Aufteilung mit 0,048 / 0,140 / 0,302 (Plan: 0,044 / 0,133 / 0,309); die exakte 50 : 50-Aufteilung gilt nur bei **bekannten** Koeffizienten.
- **Wichtigkeit in R²-Einheiten:** der Plan nannte schon die Wichtigkeit als Abfall von R² auf den Out-of-Bag-Zeilen; in den RMSE-Einheiten von `random-forest-demo` würde die Summe mit ρ steigen, weil Var(y) mitwächst.
- **Ohne Gegenstück im Plan:** Welt und Schwelle der Gruppenbildung sind Regler; ρ hat acht feste Stufen (0 bis 0,999); die Hilfetexte der Presets nennen Zahlen des Rasters und exakte Werte.

## Modell

Zwei Zwillingskennzahlen A und B (Standardnormal, Korrelation ρ) und `noise` unabhängige Rauschkennzahlen; die Zielgröße (Wartezeit) hat zwei Welten: **beide tragen bei** (y = A + B + 0,5 e) oder **nur A trägt bei** (y = 2 A + 0,5 e, B ist nur eine verrauschte Kopie). Erzeuger `kmk_algorithm.py`, Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo` (`prs_rng.py`, `prs_constants.py`, `prs_scenario.py`, `prs_algorithm.py`); der Random Forest (Regressionsbäume, Bootstrap, alle Merkmale an jedem Knoten) ist eine wörtliche Kopie aus `random-forest-demo` (`rf_tree.py`, `rf_algorithm.py`). 

## Methodik

- **Wichtigkeit = Abfall von R²:** Anstieg des mittleren Fehlerquadrats beim Mischen einer Spalte, geteilt durch Var(y), gemessen je Baum auf den **Out-of-Bag-Zeilen** und über die Bäume gemittelt (`kmk_algorithm.py`, ohne scipy zur Laufzeit). Eine Spalte wird einzeln gemischt, eine **Gruppe** (A und B) mit **einer gemeinsamen** Umordnung.
- **Bedingtes Mischen** (grobe Näherung der bedingten Wichtigkeit): A wird nur **innerhalb** von zehn Quantilklassen von B umgeordnet (und umgekehrt); die Abhängigkeit zwischen beiden bleibt erhalten, übrig bleibt, was A **zusätzlich zu B** beiträgt.
- **Extrapolation:** Anteil der Zeilen, die nach dem Mischen von A außerhalb des 3-Sigma-Bereichs der gemeinsamen Verteilung liegen (Mahalanobis-Abstand mit der wahren Korrelation); **exakt** über die Summe zweier skalierter χ²₁ (Quadratur), daneben gezählt.
- **Lineares Modell als Gegenprobe:** mit bekannten Koeffizienten ist die Wichtigkeit einer Gruppe exakt 2 b_Gᵀ Σ_GG b_G / (bᵀ Σ b + σ²); das angepasste Modell (kleinste Quadrate) wird mit derselben allgemeinen Permutationswichtigkeit gemessen wie jedes andere Modell.
- **Gruppen aus der Spearman-Korrelation:** Spalten mit |Spearman| über einer Schwelle kommen in eine Gruppe (Vereinigung über Paare).
- **Experimente:** (1) die Summe schrumpft, (2) die Aufteilung springt, (3) der Zwilling ohne Information, (4) Mischen erzeugt Unmögliches, (5) drei Antworten.
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/importance_grid.json`, 238 Sekunden auf 10 Prozessen): Zwillinge (10 Zellen: beide Welten · ρ = 0, 0,5, 0,9, 0,99, 0,999, je 4 · 25 Wälder mit n = 600, 30 Bäumen und 2 Rauschmerkmalen), angepasstes lineares Modell (5 Zellen, je 100 Datensätze), Extrapolation (5 Zellen, je 200 Datensätze), Zahl der Bäume (3 Zellen: 10, 30, 100 Bäume bei ρ = 0,99, je 6 Datensätze mit 40 Wäldern und 40 neuen Datensätzen). Vier Seeds je Zelle (101 bis 104). **Live:** ein Wald mit allen Wichtigkeiten, bis n = 1 500 und 60 Bäumen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die Tabellen Zeile für Zeile in der Rundung der App).

**Beide tragen bei: die Summe schrumpft** (Experiment 1; Mittel über 100 Wälder je ρ; Wichtigkeit = Abfall von R²; Rauschen = Mittel der Rauschmerkmale; rechts das lineare Modell mit **bekannten** Koeffizienten, Summe 4/(2,25 + 2ρ), und mit angepassten):

| ρ | A | B | Rauschen | Summe A + B | Gruppe (A, B gemeinsam) | linear exakt: Summe | linear angepasst: Summe |
|---|---|---|---|---|---|---|---|
| 0 | 0,833 | 0,838 | 0,000 | 1,671 | 1,667 | 1,778 | 1,785 |
| 0,5 | 0,611 | 0,617 | 0,000 | 1,228 | 1,773 | 1,231 | 1,232 |
| 0,9 | 0,589 | 0,636 | 0,000 | 1,225 | 1,838 | 0,988 | 0,990 |
| 0,99 | 0,696 | 0,724 | 0,000 | 1,420 | 1,869 | 0,946 | 0,966 |
| 0,999 | 0,751 | 0,707 | 0,000 | 1,458 | 1,864 | 0,942 | 1,136 |

**Die Aufteilung springt** (Experiment 2; Anteil von A an A + B, Mittel und Streuung über 100 Wälder; Anteil der Läufe, in denen einem Zwilling mehr als 70 % gehören, bei 100 Wäldern je Zelle mit einer Unsicherheit bis etwa ± 5 Prozentpunkte; Streuung im angepassten linearen Modell):

| ρ | Anteil A, Mittel | Streuung (Wald) | Läufe mit mehr als 70 : 30 | Streuung (linear, angepasst) |
|---|---|---|---|---|
| 0 | 0,498 | 0,032 | 0,0 % | 0,024 |
| 0,5 | 0,498 | 0,040 | 0,0 % | 0,023 |
| 0,9 | 0,481 | 0,143 | 17,0 % | 0,048 |
| 0,99 | 0,490 | 0,181 | 33,0 % | 0,140 |
| 0,999 | 0,515 | 0,135 | 20,0 % | 0,302 |

**Mehr Bäume beruhigen die Streuung zwischen Datensätzen kaum** (Experiment 2; ρ = 0,99; Streuung des Anteils von A, links bei gleichen Daten und neuem Wald (Mittel über 6 Datensätze mit je 40 Wäldern), rechts bei neuen Daten und je einem Wald (40 Datensätze, Rauschen etwa ± 0,02)):

| Bäume | gleiche Daten, neuer Wald | neue Daten |
|---|---|---|
| 10 | 0,091 | 0,200 |
| 30 | 0,050 | 0,173 |
| 100 | 0,028 | 0,163 |

**Der Zwilling ohne eigene Information** (Experiment 3; nur A trägt bei, B ist eine verrauschte Kopie; 100 Wälder je ρ):

| ρ | A | B | B / A |
|---|---|---|---|
| 0 | 1,866 | 0,000 | 0,000 |
| 0,5 | 1,859 | 0,000 | 0,000 |
| 0,9 | 1,844 | 0,002 | 0,001 |
| 0,99 | 1,627 | 0,080 | 0,049 |
| 0,999 | 0,992 | 0,504 | 0,508 |

**Mischen erzeugt Unmögliches** (Experiment 4; Anteil der Zeilen außerhalb des 3-Sigma-Bereichs nach dem Mischen von A, 200 Datensätze mit 600 Zeilen je ρ; exakt per Quadratur):

| ρ | gezählt | exakt |
|---|---|---|
| 0 | 1,1 % | 1,1 % |
| 0,5 | 4,3 % | 4,4 % |
| 0,9 | 35,7 % | 35,8 % |
| 0,99 | 76,8 % | 77,1 % |
| 0,999 | 92,5 % | 92,7 % |

**Drei Antworten** (Experiment 5; beide tragen bei; Wichtigkeit von A und B einzeln, beider als Gruppe, Gruppe / Summe der Einzelwerte, A bedingt auf B, B bedingt auf A, Anteil der Läufe, in denen die Spearman-Gruppierung mit der Schwelle 0,9 A und B zusammenfasst):

| ρ | A einzeln | B einzeln | Gruppe | Gruppe / Summe | A bedingt auf B | B bedingt auf A | Gruppe gefunden |
|---|---|---|---|---|---|---|---|
| 0 | 0,833 | 0,838 | 1,667 | 1,00 | 0,799 | 0,808 | 0,0 % |
| 0,5 | 0,611 | 0,617 | 1,773 | 1,44 | 0,438 | 0,439 | 0,0 % |
| 0,9 | 0,589 | 0,636 | 1,838 | 1,50 | 0,111 | 0,112 | 17,0 % |
| 0,99 | 0,696 | 0,724 | 1,869 | 1,32 | 0,040 | 0,036 | 100,0 % |
| 0,999 | 0,751 | 0,707 | 1,864 | 1,28 | 0,040 | 0,025 | 100,0 % |

| Frage | Befund |
|---|---|
| Schrumpft die Summe der Wichtigkeiten? (Experiment 1) | **Ja, bis ρ = 0,5:** 1,67 (ρ = 0) → **1,23** (ρ = 0,5); bis ρ = 0,9 bleibt sie bei 1,22 bis 1,23 und **steigt** erst ab ρ = 0,99 wieder (1,42 / 1,46 bei ρ = 0,99 / 0,999), vermutlich weil der Wald die Splits unter den Zwillingen verteilt. Die Gruppe liegt bei **1,77 bis 1,87** (ρ ≥ 0,5; ρ = 0: 1,67). Im linearen Modell mit bekannten Koeffizienten ist die Summe exakt 1,78 → 1,23 → 0,99 → 0,95 → 0,94; **der Wald trifft sie nur bei ρ = 0,5 mit 1,23** (bei ρ = 0,9: 1,22 gegen 0,99). Rauschmerkmale bleiben bei 0,00. |
| Wem gehört die Wichtigkeit? (Experiment 2) | **Mal dem einen, mal dem anderen:** das Mittel bleibt 50 : 50 (0,48 bis 0,52), die Streuung wächst von 0,032 (ρ = 0) auf **0,143 / 0,181** (ρ = 0,9 / 0,99) und liegt bei ρ = 0,999 bei 0,135; bei ρ = 0,99 gehört in etwa einem Drittel der Läufe (**33,0 %**, ± 5 Prozentpunkte) einem Zwilling mehr als 70 %. |
| Beruhigen mehr Bäume die Aufteilung? (Experiment 2) | **Nur bei gleichen Daten:** dort fällt die Streuung von 0,091 (10 Bäume) auf 0,028 (100 Bäume), bei **neuen Daten** nur von 0,200 auf 0,163: die Aufteilung hängt an der Stichprobe, nicht am Zufall des Waldes. |
| Springt das lineare Modell auch? (Experiment 2) | **Ja, wenn es angepasst wird:** Streuung der Aufteilung 0,024 (ρ = 0), 0,048 (0,9), **0,140** (0,99), **0,302** (0,999), mehr als beim Wald bei ρ = 0,999; die Koeffizienten wandern bei Multikollinearität. Mit **bekannten** Koeffizienten wäre die Aufteilung exakt 50 : 50. |
| Verwässert die Korrelation die Wichtigkeit eines Zwillings ohne Information? (Experiment 3) | **Nein, nicht bis ρ = 0,9:** B bleibt bei höchstens 0,002, A bei 1,84 bis 1,87; erst bei ρ = 0,99 bekommt B **0,080** und bei ρ = 0,999 **0,504** (A fällt von 1,87 auf 0,99, die Summe sinkt auf 1,50, die Gruppe bleibt bei 1,86 bis 1,87). |
| Was erzeugt das Mischen eines Zwillings? (Experiment 4) | Bei ρ = 0 die natürlichen **1,1 %** (e^(−4,5)), bei ρ = 0,5 **4,4 %**, bei ρ = 0,9 **35,8 %**, bei ρ = 0,99 **77,1 %**, bei ρ = 0,999 **92,7 %** der Zeilen außerhalb des 3-Sigma-Bereichs (exakt; gezählt höchstens 0,5 Prozentpunkte daneben). Das sind Zeilen aus einem Bereich, in dem nur etwa 1 % der echten Zeilen liegen; vermutlich extrapoliert der Wald dort. |
| Welche Antwort ist die richtige? (Experiment 5) | **Es sind drei Fragen:** bei ρ = 0,99 ist A einzeln **0,70**, als Gruppe mit B **1,87** und bedingt auf B **0,04**. Die Gruppe liegt mit 1,77 bis 1,87 auf dem Niveau der Summe von ρ = 0 (1,67) oder darüber; das bedingte Mischen zeigt, dass A zusätzlich zu B kaum etwas beiträgt (0,80 → 0,44 → 0,11 → 0,04). |
| Findet die Spearman-Gruppierung die Zwillinge? (Experiment 5) | **Ab ρ ≈ 0,93 in fast allen Läufen** (Schwelle 0,9, n = 600; das Raster hat nur die Stufen 0,9 und 0,99): in 0 % der Läufe bei ρ = 0 und 0,5, in **17 %** bei ρ = 0,9 (die Stichprobenkorrelation liegt meist knapp darunter) und in 100 % bei ρ = 0,99 und 0,999. Die Schwelle ist ein Regler. |
| Handwerte | Ein Modell, das nur A liest (Modell = 2 A, y = 2 A), hat die Wichtigkeit 2 (Erwartung über alle Umordnungen von vier Wochen); zwei identische Eingaben (Modell = A + B mit B = A) einzeln 0,5, gemeinsam gemischt 2,0; bei ρ = 0 ist der Anteil außerhalb von 3 Sigma e^(−4,5) = 0,0111; eine konstante Spalte und ein Modell, das sie nicht liest, haben die Wichtigkeit exakt 0. |

## Ehrliche Grenzen

- **Wichtigkeit ist keine Wirkung.** Die Zahl misst, was das **Modell** nutzt; auch eine Kopie ohne Wirkung kann Wichtigkeit bekommen (Experiment 3). Was eine Kennzahl bewirkt, klärt die Linie *Kausale Diagramme und Adjustierung*.
- **Ein Wald mit festen Einstellungen** (volle Bäume, `min_leaf` = 5, alle Merkmale an jedem Knoten): andere Bäume, Tiefen oder Modelle teilen anders auf; die Aufteilung hängt am Modell.
- **Zwei Zwillinge in zwei Welten** sind das einfachste Beispiel; echte Kennzahlen hängen in Gruppen über viele Spalten zusammen.
- **Bedingtes Mischen in zehn Klassen** ist eine grobe Näherung; Knockoffs, bedingte Verfahren mit Modellen und SHAP-Werte sind nicht gebaut.
- **Lineare Welt ohne Wechselwirkungen:** mit Wechselwirkungen ist die Wichtigkeit einer Gruppe nicht die Summe ihrer Teile.
- **Out-of-Bag-Zeilen** als Prüfdaten: bei sehr wenigen Zeilen oder Bäumen sind die Zahlen unruhig.
- **Die Wochen sind unabhängig** (Autokorrelation: Stück 4); die Daten sind simuliert, keine Aussage über echte Standorte.
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_korrelierte_merkmale.py`): die allgemeine Permutationswichtigkeit gegen **`sklearn.inspection.permutation_importance`** (gleiche Definition, Abfall von R²) und gegen die **exakte Formel des linearen Modells** (Einzelspalten und Gruppe, beide Welten, ρ bis 0,99); die Out-of-Bag-Wichtigkeit (einzeln und als Gruppe) und das bedingte Mischen gegen **Schleifenfassungen** mit demselben Zufallsstrom, die den Fehler Zeile für Zeile rechnen; der Anteil außerhalb gegen **`scipy.integrate.quad`** und gegen eine Zählung an 200 000 Paaren; die Spearman-Gruppen gegen `scipy.stats.spearmanr` und die Paarzählung.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`: Modell = 2 A, zwei identische Eingaben, konstante Spalte), Reglergrenzen, ρ an den Enden, ohne Rauschmerkmale, kleinster und größter Wald.
- **Eigenschaften** (`tests/test_eigenschaften_korrelierte_merkmale.py`): Invarianz gegen Verschieben und Strecken einer Spalte (exakt) und gegen monotone nichtlineare Umformungen (fast: die Schwellen der Bäume liegen in der Wertemitte), Austauschbarkeit der Zwillinge, Gruppe größer als jede Einzelwichtigkeit, bedingt kleiner als einzeln, Rauschspalten bei null, gleiche Seeds gleiche Zahlen.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers und des Waldes gegen die gespeicherte, jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `kmk_algorithm` (die wörtlichen Kopien `prs_*` und `rf_*` sind ausgenommen, sie werden in `pearson-demo` und `random-forest-demo` geprüft; Auswertung, Verdikte und Formen prüfen die übrigen Tests): 67 Stellen; 60 durch eigene Tests gefangen, 2 durch wortgleiche Zeilen in Nachbar-Demos, **4 als gleichwertig festgehalten** (die Schleifengrenzen und der Zählbeginn der Paarsuche der Spearman-Gruppierung, die dieselben Paare nur doppelt oder gar nicht zusätzlich prüfen, und die Spiegelung der Gauß-Legendre-Knoten bei der exakten Rechnung, deren Integrand gerade ist); **1 Stelle** (Zeile 213, die Wurzelsuche der Gruppierung) endet mit Zeitüberschreitung, der Mutant bleibt in einer Endlosschleife hängen und wird vom Werkzeug nicht gezählt. Der erste Lauf fand 8 Überlebende; 4 davon waren echte Lücken an inklusiven Grenzen (Baum mit genau `min_oob` Out-of-Bag-Zeilen im einfachen und im bedingten Mischen, die Spearman-Schwelle, der Rand des 3-Sigma-Bereichs) und führten zu Tests in `tests/test_edge_cases.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/importance_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `kmk_algorithm.py` | Kern: Zwillinge, Wichtigkeit (Out-of-Bag und allgemein), Gruppe, bedingt, Extrapolation (exakt und gezählt), lineares Modell, Spearman-Gruppen |
| `kmk_constants.py` | Regler-Grenzen, Welten, Raster der Messreihen, Farben, Presets |
| `kmk_evaluation.py` | Einzelwald, Zellen des Rasters, Verdikte |
| `kmk_layout.py` | Fester Kopf (Stylesheet) |
| `kmk_pdf_export.py` | PDF-Export der Hauptansicht |
| `kmk_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `kmk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Spearman, Konstanten) |
| `rf_tree.py`, `rf_algorithm.py` | wörtliche Kopien aus `random-forest-demo` (Baumkern, Wald mit Out-of-Bag-Zeilen) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und scikit-learn für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe, Kopie-Test |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **SHAP, Knockoffs und Verfahren der bedingten Wichtigkeit mit eigenem Modell** (Strobl u. a.): nur als Hinweis; das bedingte Mischen in Klassen ist eine grobe Näherung.
- **Merkmalsauswahl mit Strafen (Lasso)** und andere Modelle (Boosting, neuronale Netze): die Aufteilung hängt am Modell, die Demo zeigt einen Wald und ein lineares Modell.
- **Wechselwirkungen und echte Daten:** die Welten sind additiv und simuliert.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (238 Sekunden auf zehn Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Wald-Kern das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
