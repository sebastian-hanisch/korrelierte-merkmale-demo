"""Rechnet das Raster der Messreihen vor (E1 bis E5) und schreibt data/importance_grid.json.

    python tools/make_grid.py            # volles Raster, mehrere Prozesse
    python tools/make_grid.py --quick    # kleines Probe-Raster für Tests (nicht in data/importance_grid.json)

Das Raster ist deterministisch (SplitMix64 und feste Seeds, der Wald aus numpy-Generatoren mit festen Seeds); tests/test_grid.py rechnet eine feste Probe nach und vergleicht sie mit der Datei."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kmk_constants as C  # noqa: E402
import kmk_evaluation as ev  # noqa: E402

JOBS = {"twin": ev.twin_cell, "linear": ev.linear_cell, "outside": ev.outside_cell, "trees": ev.trees_cell}


def _run(task):
    kind, args = task
    return kind, JOBS[kind](*args)


def main() -> None:
    quick = "--quick" in sys.argv
    tasks = [("trees", (n,)) for n in sorted(C.TREE_COUNTS, reverse=True)]                                        # die teuren Zellen zuerst
    tasks += [("twin", (w, r)) for w in C.WORLDS for r in C.GRID_RHOS]
    tasks += [("linear", (r,)) for r in C.GRID_RHOS]
    tasks += [("outside", (r,)) for r in C.GRID_RHOS]
    if quick:
        tasks = [("twin", ("beide", 0.9, 1, (7,), 200, 8, 1)), ("linear", (0.9, 1, (7,), 200, 1, 10)), ("outside", (0.9, 5, 200)), ("trees", (8, 0.9, 1, 3, 200))]
    t0 = time.time()
    out = {k: [] for k in JOBS}
    with ProcessPoolExecutor(max_workers=10) as pool:
        for kind, row in pool.map(_run, tasks, chunksize=1):
            out[kind].append(row)
    grid = {"meta": {"seeds": list(C.GRID_SEEDS), "seconds": round(time.time() - t0)}, **out, "probe": ev.grid_probe()}
    dest = ROOT / "data" / ("importance_grid_quick.json" if quick else "importance_grid.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(grid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dest} geschrieben: " + ", ".join(f"{len(v)} {k}" for k, v in out.items()) + f", {grid['meta']['seconds']} s")


if __name__ == "__main__":
    main()
