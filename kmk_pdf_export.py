"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis, Gegenprobe, Extrapolation und Gruppen der Hauptansicht. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import kmk_constants as C
import kmk_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "ρ": "rho", "ν": "nu", "σ": "sigma",
             "²": "^2", "√": "sqrt ", "…": "...", "π": "pi", "|": "|"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Korrelierte Merkmale: wem gehört die Wichtigkeit?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Welt: {C.WORLDS[s.world]}   rho: {s.rho:g}   Zeilen n: {s.n}   Bäume: {s.trees}   Rauschmerkmale: {s.noise}   Schwelle Gruppe: {s.threshold:g}   Zufalls-Seed: {s.seed}")
    for title, verdict in (("Ergebnis", ev.verdict_main), ("Gegenprobe im linearen Modell", ev.verdict_exact), ("Mischen erzeugt Unmögliches", ev.verdict_outside), ("Gruppen aus der Spearman-Korrelation", ev.verdict_groups)):
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, verdict(a)[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Wirkung. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
