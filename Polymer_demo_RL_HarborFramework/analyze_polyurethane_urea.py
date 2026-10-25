#!/usr/bin/env python3
"""Analyze polyurethane–urea FTIR/mechanical data and predict tensile strength.

Uses only the Python standard library. The model is a standardized ridge
regression trained on formulation and FTIR measurements; other mechanical
outcomes are summarized separately and are not used as predictors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


TARGET = "tensile_strength_MPa"
MECHANICAL = [TARGET, "elongation_at_break_pct", "Youngs_modulus_MPa"]
FEATURES = [
    "hard_segment_wt_pct",
    "urea_fraction_pct",
    "nco_index",
    "soft_segment_MW",
    "post_cure_hours",
    "relative_humidity_pct",
    "ftir_amide_I_peak_cm1",
    "ftir_amide_II_peak_cm1",
    "ftir_NH_stretch_peak_cm1",
    "ftir_carbonyl_relative_intensity",
    "ftir_NH_relative_intensity",
    "ftir_ether_relative_intensity",
]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with pivoted Gauss-Jordan elimination."""
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Regression design matrix is singular.")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [augmented[i][-1] for i in range(n)]


def fit_ridge(rows: list[dict[str, str]], alpha: float) -> tuple[list[float], list[float], list[float]]:
    values = [[float(row[name]) for name in FEATURES] for row in rows]
    means = [mean([row[j] for row in values]) for j in range(len(FEATURES))]
    scales = []
    for j, center in enumerate(means):
        variance = mean([(row[j] - center) ** 2 for row in values])
        scales.append(math.sqrt(variance) or 1.0)
    design = [[1.0] + [(row[j] - means[j]) / scales[j] for j in range(len(FEATURES))] for row in values]
    targets = [float(row[TARGET]) for row in rows]
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(len(design[0]))] for i in range(len(design[0]))]
    xty = [sum(x[i] * y for x, y in zip(design, targets)) for i in range(len(design[0]))]
    for i in range(1, len(xtx)):
        xtx[i][i] += alpha
    coefficients = solve(xtx, xty)
    return means, scales, coefficients


def predict(row: dict[str, str], means: list[float], scales: list[float], coefficients: list[float]) -> float:
    return coefficients[0] + sum(
        coefficients[j + 1] * (float(row[name]) - means[j]) / scales[j]
        for j, name in enumerate(FEATURES)
    )


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    residuals = [y - p for y, p in zip(actual, predicted)]
    actual_mean = mean(actual)
    ss_res = sum(error * error for error in residuals)
    ss_tot = sum((value - actual_mean) ** 2 for value in actual)
    return {
        "n": len(actual),
        "mae_MPa": mean([abs(error) for error in residuals]),
        "rmse_MPa": math.sqrt(mean([error * error for error in residuals])),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
    }


def describe(rows: list[dict[str, str]], columns: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for column in columns:
        values = [float(row[column]) for row in rows]
        result[column] = {
            "mean": mean(values),
            "std": math.sqrt(mean([(value - mean(values)) ** 2 for value in values])),
            "min": min(values),
            "max": max(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("synthetic_polyurethane_urea.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge penalty (default: 1.0).")
    args = parser.parse_args()
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1")
    if args.alpha < 0:
        parser.error("--alpha must be non-negative")

    with args.input.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"sample_id", *FEATURES, *MECHANICAL}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("At least three complete samples are required.")
    for index, row in enumerate(rows, start=2):
        for name in [*FEATURES, *MECHANICAL]:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in row {index}, column {name}.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite value in row {index}, column {name}.")

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    n_test = max(1, min(len(rows) - 1, round(len(rows) * args.test_size)))
    test_set = set(indices[:n_test])
    train_rows = [row for i, row in enumerate(rows) if i not in test_set]
    test_rows = [row for i, row in enumerate(rows) if i in test_set]
    means, scales, coefficients = fit_ridge(train_rows, args.alpha)
    actual = [float(row[TARGET]) for row in test_rows]
    predicted = [predict(row, means, scales, coefficients) for row in test_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", "actual_tensile_strength_MPa", "predicted_tensile_strength_MPa", "error_MPa"])
        writer.writeheader()
        for row, y, p in zip(test_rows, actual, predicted):
            writer.writerow({
                "sample_id": row["sample_id"],
                "actual_tensile_strength_MPa": f"{y:.6f}",
                "predicted_tensile_strength_MPa": f"{p:.6f}",
                "error_MPa": f"{p - y:.6f}",
            })

    report = {
        "dataset": str(args.input),
        "n_samples": len(rows),
        "split": {"train": len(train_rows), "test": len(test_rows), "test_fraction": len(test_rows) / len(rows), "random_seed": args.seed},
        "model": {"name": "standardized_ridge_regression", "alpha": args.alpha, "target": TARGET, "features": FEATURES},
        "test_metrics": metrics(actual, predicted),
        "descriptive_statistics": describe(rows, [*FEATURES[6:], *MECHANICAL]),
        "prediction_file": str(predictions_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = args.output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Trained on {len(train_rows)} samples; evaluated on {len(test_rows)} held-out samples.")
    print(f"MAE: {report['test_metrics']['mae_MPa']:.3f} MPa | RMSE: {report['test_metrics']['rmse_MPa']:.3f} MPa | R²: {report['test_metrics']['r2']:.3f}")
    print(f"Wrote {predictions_path} and {metrics_path}")


if __name__ == "__main__":
    main()
