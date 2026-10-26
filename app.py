"""Korrelierte Merkmale: wem gehört die Wichtigkeit? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zehntes Stück der Korrelations-Reihe. Zwei Kennzahlen A und B messen fast dasselbe (Korrelation ρ); ein Random Forest sagt die Wartezeit voraus. Gezeigt werden die Permutationswichtigkeit
(Abfall von R² beim Mischen), die springende Aufteilung zwischen den Zwillingen, das Mischen von Punkten außerhalb des Datenbereichs und drei Gegenmittel (Gruppe mischen,
bedingt mischen, Gruppen aus der Spearman-Korrelation).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kmk_constants as C
import kmk_evaluation as ev
from kmk_layout import dunkles_design, kopf_css
from kmk_pdf_export import build_pdf
from kmk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kmk_visualization import (
    build_importance_bars,
    build_only_a,
    build_outside,
    build_remedies,
    build_spread,
    build_sum_lines,
    build_trees_chart,
    build_twin_scatter,
)

st.set_page_config(page_title="Korrelierte Merkmale – Sebastian Hanisch", layout="wide")


def de(v: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{v:.{digits}f}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kmk_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🪞 Korrelierte Merkmale: wem gehört die Wichtigkeit?")
st.markdown(
    """
**Zehntes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios mit Wichtigkeit unter Korrelation der Merkmale**. Zwei Kennzahlen eines Standorts, **A und B**, messen fast dasselbe
(zum Beispiel die Auslastung zweier Messpunkte an derselben Rampe); ein **Random Forest** sagt daraus die Wartezeit voraus, und die **Permutationswichtigkeit** sagt, welche Kennzahl „wichtig“ ist.
Die Frage der Stücke 1 bis 9 war, was eine Zahl über zwei Größen sagt. Hier steht die Korrelation **zwischen den Merkmalen** im Weg: Wem gehört die Wichtigkeit, wenn A und B dasselbe wissen?

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Die Wichtigkeit bei **unabhängigen** Rauschmerkmalen (Gini gegen Permutation) zeigt
`random-forest-demo`, dekorrelierte Bäume `feature-bagging-demo`; **was ein Merkmal bewirkt**, steht in der Linie *Kausale Diagramme und Adjustierung*: Wichtigkeit ist keine Wirkung.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Zwillinge A und B** | Zwei Kennzahlen mit der Korrelation ρ. In der Welt „beide tragen bei“ gilt Wartezeit = A + B + Rauschen, in der Welt „nur A trägt bei“ ist B nur eine verrauschte Kopie von A. Dazu kommen unabhängige Rauschkennzahlen. |
| **Permutationswichtigkeit** | Eine Spalte wird zufällig durcheinandergemischt; um wie viel schlechter sagt der Wald danach voraus? Angegeben als **Abfall von R²** auf den Zeilen, die ein Baum nie gesehen hat (Out-of-Bag). |
| **Einzeln mischen** | Nur A (oder nur B) wird gemischt. Bei starker Korrelation entstehen dabei Zeilen aus einem Bereich, in dem nur wenige echte Zeilen liegen (**Extrapolation**). |
| **Gruppe mischen** | A und B werden mit **derselben** Umordnung gemischt und bleiben zusammen: die Wichtigkeit der Information, die beide gemeinsam tragen. |
| **Bedingt mischen** | A wird nur innerhalb von zehn Klassen von B gemischt (und umgekehrt): was A **zusätzlich zu B** beiträgt. Grobe Näherung der „bedingten Wichtigkeit“. |
| **Gruppen aus der Spearman-Korrelation** | Spalten mit \|Spearman\| über einer Schwelle kommen in eine Gruppe; die Schwelle ist ein Regler, kein Naturgesetz. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Daten**")
    world = st.selectbox("Welt", list(C.WORLDS), format_func=C.WORLDS.get, key="world_select",
                         help="Trägt nur A zur Wartezeit bei (B ist eine verrauschte Kopie), oder tragen beide bei?")
    rho = st.select_slider("Korrelation ρ der Zwillinge", options=C.RHOS, key="rho_select", format_func=lambda v: f"{v:g}".replace(".", ","),
                           help="Wie ähnlich A und B sind. Bei 0,99 messen sie praktisch dasselbe.")
    n = st.slider("Zeilen n", *bounds("n_slider"), key="n_slider", help="Zahl der Wochen (Zeilen) der Trainingsdaten des Waldes.")
    noise = st.slider("Rauschmerkmale", *bounds("noise_slider"), key="noise_slider", help="Unabhängige Kennzahlen ohne Wirkung; ihre Wichtigkeit sollte bei 0 liegen.")
    st.markdown("**Verfahren**")
    trees = st.slider("Bäume", *bounds("trees_slider"), key="trees_slider", help="Zahl der Bäume des Random Forest.")
    threshold = st.slider("Schwelle der Gruppenbildung (|Spearman|)", *bounds("threshold_slider"), step=0.01, key="threshold_slider", format="%.2f",
                          help="Spalten mit einer Rangkorrelation über dieser Schwelle werden zu einer Gruppe.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Datensatz ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"world_select": world, "rho_select": float(rho), "n_slider": int(n), "trees_slider": int(trees), "noise_slider": int(noise), "threshold_slider": float(threshold), "seed_input": int(seed)})

settings = ev.Settings(world, float(rho), int(n), int(trees), int(noise), float(threshold), int(seed))
with st.spinner("Der Wald wird angepasst …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
ia, ib = float(a.singles[0]), float(a.singles[1])
cols = st.columns(3)
cols[0].metric("A einzeln gemischt", de(ia), help="Abfall von R², wenn nur A gemischt wird (Out-of-Bag-Zeilen).")
cols[1].metric("B einzeln gemischt", de(ib), help="Abfall von R², wenn nur B gemischt wird.")
cols[2].metric("Anteil von A an A + B", f"{100 * ia / (ia + ib):.0f} %" if ia + ib > 0 else "–", help="Springt von Datensatz zu Datensatz, wenn A und B dasselbe wissen (Experiment 2).")
cols = st.columns(3)
cols[0].metric("A + B als Gruppe", de(a.group), help="Beide Spalten mit derselben Umordnung gemischt.")
cols[1].metric("A bedingt auf B", de(a.cond_a), help="A nur innerhalb von zehn Klassen von B gemischt: was A zusätzlich beiträgt.")
cols[2].metric("Außerhalb nach dem Mischen", f"{100 * a.outside:.0f} %", help="Anteil der Zeilen, die nach dem Mischen von A außerhalb des 3-Sigma-Bereichs von A und B liegen.")
left, right = st.columns(2)
with left:
    st.plotly_chart(build_twin_scatter(a.X, a.shuffled_a, settings.rho, mixed=False), width="stretch", key="main_real")
    st.caption("Die echten Wochen: A und B liegen entlang der Diagonalen. Gestrichelt: der 3-Sigma-Bereich der gemeinsamen Verteilung.")
with right:
    st.plotly_chart(build_twin_scatter(a.X, a.shuffled_a, settings.rho, mixed=True), width="stretch", key="main_mixed")
    st.caption("Dieselben Wochen, nachdem A gemischt wurde: Punkte außerhalb des Bereichs (rot) gibt es in der Welt nicht, der Wald muss dort raten.")
kind, text = ev.verdict_outside(a)
getattr(st, kind)(text)
st.plotly_chart(build_importance_bars(a), width="stretch", key="main_bars")
st.caption("Wichtigkeit als Abfall von R² auf den Out-of-Bag-Zeilen: Zwillinge einzeln gemischt, Rauschmerkmale, die Gruppe aus A und B und die bedingten Werte.")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)
kind, text = ev.verdict_exact(a)
getattr(st, kind)(text)
kind, text = ev.verdict_groups(a)
getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="korrelierte-merkmale-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wem gehört die Wichtigkeit?")
st.markdown("Bei starker Korrelation **niemandem eindeutig**: einzeln gemischt teilen sich die Zwillinge sie von Datensatz zu Datensatz anders auf, als Gruppe gehört sie beiden gemeinsam, und bedingt auf den Partner bleibt für jeden fast nichts (Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/importance_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")

# --- E1 Die Summe schrumpft -----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Summe der Wichtigkeiten schrumpft")
st.caption(f"Welt „beide tragen bei“, n = {C.GRID_N}, {C.GRID_TREES} Bäume, {C.GRID_NOISE} Rauschmerkmale. Mittel über {4 * C.GRID_RUNS} Wälder je ρ (vier Seeds). "
           "Gestrichelt: das lineare Modell mit den **wahren** Koeffizienten (exakte Formel), gepunktet: das angepasste lineare Modell.")
if grid is not None:
    st.plotly_chart(build_sum_lines(grid["twin"], grid["linear"]), width="stretch", key="sum_chart")
    kind, text = ev.verdict_sum_grid(grid["twin"], grid["linear"])
    getattr(st, kind)(text)
    st.caption("Je ähnlicher die Zwillinge, desto weniger **zusätzliche** Information trägt jeder einzeln: mischt man nur einen, kennt der Wald den anderen noch. Die Summe der Einzelwerte liegt bei ρ > 0 unter der bei ρ = 0, "
               "die Gruppe (beide gemeinsam gemischt) nicht.")

st.markdown("---")

# --- E2 Die Aufteilung springt -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Aufteilung zwischen den Zwillingen springt")
st.caption("Anteil von A an A + B; links die Streuung von Datensatz zu Datensatz, rechts die Streuung bei gleichen Daten (neuer Wald) und bei neuen Daten, je nach Zahl der Bäume (ρ = 0,99).")
if grid is not None:
    e2_rho = st.selectbox("ρ für das Verdikt", list(C.GRID_RHOS), index=3, key="e2_rho", format_func=lambda v: f"{v:g}".replace(".", ","))
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(build_spread(grid["twin"], grid["linear"]), width="stretch", key="spread_chart")
    with s2:
        st.plotly_chart(build_trees_chart(grid["trees"]), width="stretch", key="trees_chart")
    kind, text = ev.verdict_spread(grid["twin"], grid["linear"], e2_rho)
    getattr(st, kind)(text)
    kind, text = ev.verdict_trees(grid["trees"])
    getattr(st, kind)(text)
    st.caption("Das Mittel der Aufteilung bleibt bei 50 : 50 (die Zwillinge sind austauschbar), aber in einem einzelnen Datensatz gehört oft einem von beiden fast alles. Auch das angepasste lineare Modell springt, "
               "weil seine Koeffizienten bei Multikollinearität wandern; nur mit **bekannten** Koeffizienten ist die Aufteilung exakt 50 : 50.")

st.markdown("---")

# --- E3 Der Zwilling ohne Information -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Der Zwilling ohne eigene Information")
st.caption("Nur A trägt zur Wartezeit bei, B ist eine verrauschte Kopie von A. Wichtigkeit von A und B gegen ρ.")
if grid is not None:
    st.plotly_chart(build_only_a(grid["twin"]), width="stretch", key="only_a_chart")
    kind, text = ev.verdict_only_a(grid["twin"])
    getattr(st, kind)(text)
    st.caption("Die Faustregel „korrelierte Merkmale verwässern die Wichtigkeit“ gilt bis ρ = 0,9 nur, wenn beide Zwillinge Information tragen. Eine bloße Kopie bekommt bis ρ = 0,9 nichts; "
               "ab ρ = 0,99 verteilt der Wald die Wichtigkeit auch auf sie.")

st.markdown("---")

# --- E4 Mischen erzeugt Unmögliches ---------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Mischen erzeugt Zeilen, die kaum vorkommen")
st.caption(f"Anteil der Zeilen, die nach dem Mischen von A außerhalb des {C.K_SIGMA:g}-Sigma-Bereichs der gemeinsamen Verteilung von A und B liegen; {C.OUTSIDE_SETS} Datensätze je ρ, daneben der exakte Wert (Quadratur).")
if grid is not None:
    e4_rho = st.selectbox("ρ für das Verdikt", list(C.GRID_RHOS), index=3, key="e4_rho", format_func=lambda v: f"{v:g}".replace(".", ","))
    st.plotly_chart(build_outside(grid["outside"]), width="stretch", key="outside_chart")
    kind, text = ev.verdict_outside_grid(grid["outside"], e4_rho)
    getattr(st, kind)(text)
    st.caption("Bei unabhängigen Merkmalen sind es nur die natürlichen 1,1 %. Bei ρ = 0,99 sind rund drei von vier gemischten Zeilen Punkte aus einem Bereich, in dem nur etwa 1 % der echten Zeilen liegen "
               "(Vermutung: dort extrapoliert der Wald, und die Wichtigkeit misst zum Teil, wie er außerhalb der Daten rät).")

st.markdown("---")

# --- E5 Drei Antworten --------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Drei Antworten auf „wie wichtig ist A?“")
st.caption("A einzeln gemischt, A und B als Gruppe gemischt, A bedingt auf B (in zehn Klassen). Welt „beide tragen bei“, Mittel über 100 Wälder je ρ.")
if grid is not None:
    e5_rho = st.selectbox("ρ für das Verdikt", list(C.GRID_RHOS), index=3, key="e5_rho", format_func=lambda v: f"{v:g}".replace(".", ","))
    st.plotly_chart(build_remedies(grid["twin"]), width="stretch", key="remedies_chart")
    kind, text = ev.verdict_remedies(grid["twin"], e5_rho)
    getattr(st, kind)(text)
    st.caption("Die Gruppe sagt, wie viel A und B **zusammen** wissen: sie bleibt bei höheren ρ stabil. Bedingt gemischt bleibt für A fast nichts, wenn B dasselbe weiß. Welche Antwort die richtige ist, "
               "hängt von der Frage ab: Was geht verloren, wenn die Kennzahl fehlt (bedingt), oder wenn beide fehlen (Gruppe)?")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Wichtigkeit heißt Wirkung** | nein: die Zahl misst, was das **Modell** nutzt, nicht was die Kennzahl **bewirkt**; auch ein Zwilling ohne Wirkung kann Wichtigkeit bekommen (Experiment 3) | Linie *Kausale Diagramme und Adjustierung* |
| **Ein Wald mit festen Einstellungen** | andere Bäume, Tiefen oder Modelle (Boosting, lineare Modelle) teilen anders auf; die Aufteilung hängt am Modell | – |
| **Zwei Zwillinge, zwei Welten** | echte Kennzahlen hängen in Gruppen zusammen, oft über viele Spalten; die Demo zeigt das einfachste Beispiel | – |
| **Bedingtes Mischen in zehn Klassen** | eine grobe Näherung der bedingten Wichtigkeit; Knockoffs und SHAP-Werte sind nicht gebaut | – |
| **Lineare Welt, keine Wechselwirkungen** | mit Wechselwirkungen ist die Wichtigkeit einer Gruppe nicht die Summe ihrer Teile | – |
| **Out-of-Bag-Zeilen als Prüfdaten** | bei sehr wenigen Zeilen oder Bäumen sind die Zahlen unruhig (kleine Bäume, wenige Out-of-Bag-Zeilen) | Zeilen und Bäume erhöhen |
| **Die Daten sind unabhängig über die Wochen** | Autokorrelation ändert die Aufteilung (Stück 4) | Stück 4 |
"""
)
st.caption("Alle Daten sind simuliert; Zwillinge und Wartezeit sind Modellwahlen, kein Befund über echte Standorte.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** $X = (A, B, R_1, \dots)$ sind die Kennzahlen, $y$ die Wartezeit, $f$ der Random Forest; $\rho$ ist die Korrelation der Zwillinge $A, B$; $\pi_G$ eine zufällige Umordnung der Zeilen.

| Symbol | Bedeutung |
|---|---|
| $G$ | Gruppe von Spalten (eine Spalte oder $\{A, B\}$) |
| $X^{\pi_G}$ | Daten, in denen die Spalten aus $G$ mit **einer** gemeinsamen Umordnung gemischt sind |
| $\operatorname{Var}(y)$ | Varianz der Zielgröße |

**Wichtigkeit (Abfall von $R^2$)** einer Gruppe $G$, gemessen auf den Out-of-Bag-Zeilen jedes Baums und gemittelt:

$$ I_G = \frac{\operatorname{E}\big[(y - f(X^{\pi_G}))^2\big] - \operatorname{E}\big[(y - f(X))^2\big]}{\operatorname{Var}(y)} $$

**Lineares Modell mit bekannten Koeffizienten.** Für $y = b^\top X + \sigma e$ ersetzt das Mischen die Gruppe durch eine unabhängige Kopie, der Fehler steigt um $2\, b_G^\top \Sigma_{GG}\, b_G$:

$$ I_G = \frac{2\, b_G^\top \Sigma_{GG}\, b_G}{b^\top \Sigma\, b + \sigma^2} $$

Für $b = (1, 1)$ und $\sigma = 0{,}5$ ist jede Zwillingsspalte einzeln $\dfrac{2}{2{,}25 + 2\rho}$ wichtig, die Gruppe $\dfrac{4(1+\rho)}{2{,}25 + 2\rho}$.

**Bedingtes Mischen.** Mit den Klassen $c(\cdot)$ der Quantile von $B$ wird $A$ nur innerhalb gleicher Klassen umgeordnet: $A^{\pi}_i = A_{\pi(i)}$ mit $c(B_{\pi(i)}) = c(B_i)$.

**Anteil außerhalb des Datenbereichs.** Nach dem Mischen von $A$ sind $A'$ und $B$ unabhängig; der Mahalanobis-Abstand mit der wahren Korrelation ist

$$ Q = \frac{A'^2 - 2\rho A' B + B^2}{1 - \rho^2} \;\sim\; \frac{U^2}{1-\rho} + \frac{V^2}{1+\rho}, \qquad U, V \text{ unabhängig } N(0,1) $$

und gesucht ist $P(Q > k^2)$ mit $k = 3$; für $\rho = 0$ ist das $e^{-k^2/2} = 0{,}0111$.

**Aufteilung** der Einzelwichtigkeiten auf die Zwillinge: $s_A = I_A / (I_A + I_B)$.

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests gegen scikit-learn, scipy und die exakte Formel des linearen Modells) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
