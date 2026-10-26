"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_rows():
    """Vier Wochen, drei Kennzahlen: A = (1, 2, 3, 4), B = A (identische Kopie), C = (5, 5, 5, 5) (konstant); Zielgröße y = 2 A.
    Mittel von A: 2,5; Var(A) = 1,25 (Teilen durch n); Var(y) = 5. Ein Modell, das nur A liest (predict = 2 A), hat den Fehler 0; ein Modell, das C liest, ändert sich nie."""
    A = np.array([1.0, 2.0, 3.0, 4.0])
    return np.column_stack([A, A, np.full(4, 5.0)]), 2.0 * A


@pytest.fixture
def twins():
    """Zwillinge mit fester Korrelation 0,5 und 2 Rauschmerkmalen, 800 Zeilen, Welt „beide“ (nur die Daten, ohne Wald)."""
    import kmk_algorithm as K
    import prs_scenario as S
    return K.draw_twins(800, 0.5, "beide", 2, S.VRng(11))
