#!/usr/bin/env python3
"""Check the recorded Gotham evaluation outputs without rerunning models."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def check(condition, description):
    if not condition:
        raise AssertionError(description)
    print(f"PASS: {description}")


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = ROOT / "checksums.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        target = ROOT / rel
        check(target.is_file() and sha256(target) == expected, f"checksum: {rel}")

    metrics = rows(RESULTS / "metrics.csv")
    expected_protocol_folds = {"P1-random": 5, "P2-device": 8, "P3-segment": 3}
    regimes = {"F-ALL", "F-NOID", "F-BEHAV"}
    models = {"logreg", "rf", "histgb", "mlp"}
    expected = {(p, r, m, f) for p, n in expected_protocol_folds.items()
                for r in regimes for m in models for f in range(n)}
    actual = {(r["protocol"], r["regime"], r["model"], int(r["fold"]))
              for r in metrics}
    check(len(metrics) == len(actual) == len(expected) == 192 and actual == expected,
          "192 unique model/regime/fold records")

    headline = json.loads((RESULTS / "headline.json").read_text())
    check(int(headline["rows"]) == 404324 and int(headline["n_devices"]) == 78,
          "working benchmark counts in run summary")
    grouped = {}
    for row in metrics:
        grouped.setdefault((row["protocol"], row["regime"], row["model"]), []).append(
            float(row["macro_f1"]))
    p1 = np.mean(grouped[("P1-random", "F-ALL", "rf")])
    p2 = np.mean(grouped[("P2-device", "F-BEHAV", "rf")])
    check(abs(p1 - float(headline["macroF1_P1_FALL"])) < 1e-4 and
          abs(p2 - float(headline["macroF1_P2_FBEHAV"])) < 1e-4,
          "headline macro-F1 values match recorded folds")
    check(abs((p1-p2)-float(headline["delta_optimism"])) < 1e-4,
          "optimism gap equals difference of fold means")

    archive = RESULTS / "confusion_matrices.zip"
    check(zipfile.is_zipfile(archive), "confusion matrix archive can be opened")
    pooled = None
    labels = None
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        check(len(names) == len(set(names)) == 192 and z.testzip() is None,
              "192 intact matrices in confusion archive")
        expected_names = {f"confusion/{p}_{r}_{m}_f{f}.npz"
                          for p, n in expected_protocol_folds.items()
                          for r in regimes for m in models for f in range(n)}
        check(set(names) == expected_names, "matrix filenames match metric records")
        per_fold_test = {(r["protocol"], int(r["fold"])): int(r["n_test"])
                         for r in metrics}
        for name in names:
            stem = name.split("/")[-1].removesuffix(".npz")
            p, r, m, fold = stem.split("_")
            fold = int(fold.removeprefix("f"))
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as data:
                matrix = data["cm"]
                current_labels = data["classes"].tolist()
                if labels is None:
                    labels = current_labels
                if current_labels != labels or matrix.shape != (7, 7):
                    raise AssertionError(f"class labels or matrix shape mismatch: {name}")
                if int(matrix.sum()) != per_fold_test[(p, fold)]:
                    raise AssertionError(f"test support mismatch: {name}")
                if (p, r, m) == ("P2-device", "F-BEHAV", "rf"):
                    if pooled is None:
                        pooled = np.zeros_like(matrix)
                    pooled += matrix
    check(int(pooled.sum()) == 404324, "pooled P2 device-disjoint test coverage")
    pooled_csv = rows(RESULTS / "tables" / "pooled_p2_behav_rf_per_class.csv")
    check(len(pooled_csv) == len(labels) == 7, "seven pooled label classes")
    for i, label in enumerate(labels):
        record = pooled_csv[i]
        total = int(pooled[i].sum())
        precision = pooled[i,i]/pooled[:,i].sum() if pooled[:,i].sum() else 0.0
        recall = pooled[i,i]/total if total else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        check(record["class"] == label and int(record["support"]) == total and
              abs(float(record["precision"]) - precision) < 1e-6 and
              abs(float(record["recall"]) - recall) < 1e-6 and
              abs(float(record["f1"]) - f1) < 1e-6,
              f"pooled precision/recall/F1: {label}")

    print(f"\nVerified headline results: P1 F-ALL RF {p1:.4f}; "
          f"P2 F-BEHAV RF {p2:.4f}; gap {p1-p2:.4f}.")
    print("Offline record verification complete. This is not a full training re-run.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
