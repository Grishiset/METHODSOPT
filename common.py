"""Shared result files and capture of the native HiGHS console log."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def records(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verified(results, tolerance=1e-7):
    for label, residual in results.items():
        if not np.isfinite(residual) or abs(residual) > tolerance:
            raise ValueError(f"{label}: {residual} exceeds {tolerance}")
    return {key: float(value) for key, value in results.items()}


def run(script, generate, experiment):
    folder = Path(script).resolve().parent
    datafile = folder / "data.json"
    out = folder / "results"
    out.mkdir(exist_ok=True)
    if sys.argv[1:] == ["--generate"]:
        dump(datafile, generate())
        return
    if sys.argv[1:] == ["--worker"]:
        experiment(json.loads(datafile.read_text(encoding="utf-8")), out)
        return
    if sys.argv[1:]:
        raise SystemExit("Options: --generate")
    with (out / "solver.log").open("wb") as log:
        subprocess.run([sys.executable, str(Path(script).resolve()), "--worker"],
                       stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print(f"Готово: {out}")
