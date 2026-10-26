"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kmk_constants as C

AXIS_LIM = 4.5


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def ellipse_points(rho: float, k: float = C.K_SIGMA, steps: int = 120) -> tuple[list[float], list[float]]:
    """Rand des k-Sigma-Bereichs der gemeinsamen Verteilung zweier Standardnormalgrößen mit Korrelation rho (Mahalanobis-Abstand k)."""
    t = np.linspace(0.0, 2.0 * math.pi, steps)
    u = k * math.sqrt(1.0 + rho) * np.cos(t) / math.sqrt(2.0)
    v = k * math.sqrt(1.0 - rho) * np.sin(t) / math.sqrt(2.0)
    return (u + v).tolist(), (u - v).tolist()


def build_twin_scatter(X, shuffled_a, rho: float, mixed: bool, height: int = 380):
    """Streudiagramm der Zwillinge A und B: die echten Wochen (blau) oder dieselben Wochen nach dem Mischen von A (rot, Punkte außerhalb des Datenbereichs kräftig), dazu der 3-Sigma-Bereich."""
    X = np.asarray(X, dtype=float)
    b = X[:, 1]
    fig = go.Figure()
    ex, ey = ellipse_points(rho)
    if not mixed:
        fig.add_trace(go.Scatter(x=X[:, 0].tolist(), y=b.tolist(), mode="markers", name="echte Wochen", marker=dict(size=5, color=C.COLOR_A, opacity=0.45), hovertemplate="A %{x:.2f}, B %{y:.2f}<extra></extra>"))
    else:
        a = np.asarray(shuffled_a, dtype=float)
        q = (a * a - 2.0 * rho * a * b + b * b) / (1.0 - rho * rho)
        out = q > C.K_SIGMA ** 2
        fig.add_trace(go.Scatter(x=a[~out].tolist(), y=b[~out].tolist(), mode="markers", name="gemischt, im Datenbereich", marker=dict(size=5, color=C.COLOR_NOISE, opacity=0.45),
                                 hovertemplate="A %{x:.2f}, B %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=a[out].tolist(), y=b[out].tolist(), mode="markers", name="gemischt, außerhalb (gibt es nicht)", marker=dict(size=7, color="#d62728", opacity=0.8),
                                 hovertemplate="A %{x:.2f}, B %{y:.2f}<extra>außerhalb</extra>"))
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name=f"{C.K_SIGMA:g}-Sigma-Bereich", line=dict(color=C.COLOR_EDGE, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_xaxes(title_text="Kennzahl A (normiert)", range=[-AXIS_LIM, AXIS_LIM])
    fig.update_yaxes(title_text="Kennzahl B (normiert)", range=[-AXIS_LIM, AXIS_LIM], scaleanchor="x")
    return _base(fig, height)


def build_importance_bars(a, height: int = 340):
    """Wichtigkeit der Hauptansicht: einzeln gemischt (A, B, Rauschen), als Gruppe und bedingt auf den Partner."""
    labels = ["A einzeln", "B einzeln"] + [f"Rauschen {i + 1}" for i in range(len(a.singles) - 2)] + ["A + B als Gruppe", "A bedingt auf B", "B bedingt auf A"]
    values = [float(v) for v in a.singles] + [a.group, a.cond_a, a.cond_b]
    colors = [C.COLOR_A, C.COLOR_B] + [C.COLOR_NOISE] * (len(a.singles) - 2) + [C.COLOR_GROUP, C.COLOR_COND, C.COLOR_COND]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f}".replace(".", ",") for v in values], textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Wichtigkeit (Abfall von R²)", rangemode="tozero")
    fig.update_xaxes(type="category")
    return _base(fig, height)


def _cells(rows, **keys):
    return sorted((r for r in rows if all(r[k] == v for k, v in keys.items())), key=lambda r: r["rho"])


def build_sum_lines(twin_rows, linear_rows, height: int = 360):
    """E1: Summe der Einzelwichtigkeiten, Gruppe und lineares Modell gegen rho (Welt „beide“)."""
    cells = _cells(twin_rows, world="beide")
    xs = [r["rho"] for r in cells]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["sum"] for r in cells], mode="lines+markers", name="Wald: Summe der Einzelwerte", line=dict(color=C.COLOR_FOREST, width=3), hovertemplate="ρ = %{x}: %{y:.2f}<extra>Summe</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["group"] for r in cells], mode="lines+markers", name="Wald: Gruppe A + B", line=dict(color=C.COLOR_GROUP, width=3), hovertemplate="ρ = %{x}: %{y:.2f}<extra>Gruppe</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["exact_a"] + r["exact_b"] for r in cells], mode="lines+markers", name="linear, bekannte Koeffizienten: Summe (exakt)", line=dict(color=C.COLOR_LINEAR, width=2, dash="dash"),
                             hovertemplate="ρ = %{x}: %{y:.2f}<extra>exakt</extra>"))
    lin = sorted(linear_rows, key=lambda r: r["rho"])
    fig.add_trace(go.Scatter(x=[r["rho"] for r in lin], y=[r["sum"] for r in lin], mode="lines+markers", name="linear, angepasst: Summe", line=dict(color=C.COLOR_LINEAR, width=2, dash="dot"),
                             hovertemplate="ρ = %{x}: %{y:.2f}<extra>angepasst</extra>"))
    fig.update_xaxes(title_text="Korrelation ρ der Zwillinge", type="category")
    fig.update_yaxes(title_text="Wichtigkeit (Abfall von R²)", rangemode="tozero")
    return _base(fig, height)


def build_spread(twin_rows, linear_rows, height: int = 340):
    """E2: Streuung des Anteils von A (Wald und angepasstes lineares Modell) gegen rho."""
    cells = _cells(twin_rows, world="beide")
    lin = sorted(linear_rows, key=lambda r: r["rho"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["rho"] for r in cells], y=[r["share_sd"] for r in cells], mode="lines+markers", name="Wald", line=dict(color=C.COLOR_FOREST, width=3), hovertemplate="ρ = %{x}: %{y:.3f}<extra>Wald</extra>"))
    fig.add_trace(go.Scatter(x=[r["rho"] for r in lin], y=[r["share_sd"] for r in lin], mode="lines+markers", name="lineares Modell, angepasst", line=dict(color=C.COLOR_LINEAR, width=3, dash="dot"),
                             hovertemplate="ρ = %{x}: %{y:.3f}<extra>linear</extra>"))
    fig.update_xaxes(title_text="Korrelation ρ der Zwillinge", type="category")
    fig.update_yaxes(title_text="Streuung des Anteils von A (von Datensatz zu Datensatz)", rangemode="tozero")
    return _base(fig, height)


def build_trees_chart(rows, height: int = 340):
    """E2: Streuung der Aufteilung gegen die Zahl der Bäume: bei gleichen Daten (neue Wälder) und bei neuen Daten."""
    rows = sorted(rows, key=lambda r: r["trees"])
    xs = [str(r["trees"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["same_data_sd"] for r in rows], name="gleiche Daten, neuer Wald", marker_color=C.COLOR_GROUP, hovertemplate="%{x} Bäume: %{y:.3f}<extra>gleiche Daten</extra>"))
    fig.add_trace(go.Bar(x=xs, y=[r["new_data_sd"] for r in rows], name="neue Daten", marker_color=C.COLOR_B, hovertemplate="%{x} Bäume: %{y:.3f}<extra>neue Daten</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Zahl der Bäume", type="category")
    fig.update_yaxes(title_text="Streuung des Anteils von A", rangemode="tozero")
    return _base(fig, height)


def build_only_a(twin_rows, height: int = 340):
    """E3: Wichtigkeit von A und B gegen rho, wenn nur A zur Zielgröße beiträgt."""
    cells = _cells(twin_rows, world="nur_a")
    xs = [r["rho"] for r in cells]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["a"] for r in cells], mode="lines+markers", name="A (trägt bei)", line=dict(color=C.COLOR_A, width=3), hovertemplate="ρ = %{x}: %{y:.2f}<extra>A</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["b"] for r in cells], mode="lines+markers", name="B (verrauschte Kopie)", line=dict(color=C.COLOR_B, width=3), hovertemplate="ρ = %{x}: %{y:.3f}<extra>B</extra>"))
    fig.update_xaxes(title_text="Korrelation ρ der Zwillinge", type="category")
    fig.update_yaxes(title_text="Wichtigkeit (Abfall von R²)", rangemode="tozero")
    return _base(fig, height)


def build_outside(rows, height: int = 340):
    """E4: Anteil der Zeilen außerhalb des Datenbereichs nach dem Mischen von A: gezählt (Balken) und exakt (Rauten)."""
    rows = sorted(rows, key=lambda r: r["rho"])
    xs = [str(r["rho"]).replace(".", ",") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[100 * r["measured"] for r in rows], name="gezählt", marker_color=C.COLOR_B, hovertemplate="ρ = %{x}: %{y:.1f} %<extra>gezählt</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["exact"] for r in rows], mode="markers", name="exakt", marker=dict(size=12, symbol="diamond-open", color=C.COLOR_EDGE, line=dict(width=2)), hovertemplate="ρ = %{x}: %{y:.1f} %<extra>exakt</extra>"))
    fig.update_xaxes(title_text="Korrelation ρ der Zwillinge", type="category")
    fig.update_yaxes(title_text=f"außerhalb des {C.K_SIGMA:g}-Sigma-Bereichs (%)", range=[0, 100])
    return _base(fig, height)


def build_remedies(twin_rows, height: int = 340):
    """E5: Wichtigkeit von A einzeln, als Gruppe und bedingt gegen rho (Welt „beide“)."""
    cells = _cells(twin_rows, world="beide")
    xs = [str(r["rho"]).replace(".", ",") for r in cells]
    fig = go.Figure()
    for key, name, color in (("a", "A einzeln", C.COLOR_A), ("group", "A + B als Gruppe", C.COLOR_GROUP), ("cond_a", "A bedingt auf B", C.COLOR_COND)):
        fig.add_trace(go.Bar(x=xs, y=[r[key] for r in cells], name=name, marker_color=color, hovertemplate="ρ = %{x}: %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Korrelation ρ der Zwillinge", type="category")
    fig.update_yaxes(title_text="Wichtigkeit (Abfall von R²)", rangemode="tozero")
    return _base(fig, height)
