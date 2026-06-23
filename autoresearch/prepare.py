"""Fixed task preparation and evaluation utilities.

Do not modify this file during the AutoResearch experiment loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "codesubset"

DEFAULT_SEED = 42
DEFAULT_VALID_FRACTION = 0.2
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class DataPaths:
    train_h5: Path = DATA_ROOT / "train.h5"
    train_csv: Path = DATA_ROOT / "train.csv"
    records: Path = DATA_ROOT / "train" / "RECORDS.txt"


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    valid: list[int]


@dataclass(frozen=True)
class PreparedTask:
    traces: np.ndarray
    labels: np.ndarray
    split: SplitIndices
    paths: DataPaths


def _missing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def parse_record_id(record: str) -> int:
    """Extract the numeric exam id from a RECORDS entry such as TNMG100046."""
    stem = Path(record.strip()).stem
    if "TNMG" not in stem:
        raise ValueError(f"Cannot parse exam id from record entry: {record!r}")
    return int(stem.split("TNMG", 1)[1])


def make_split_indices(
    n_items: int,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    seed: int = DEFAULT_SEED,
) -> SplitIndices:
    if n_items < 2:
        raise ValueError("Need at least two examples to create a train/validation split.")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    n_valid = int(round(n_items * valid_fraction))
    n_valid = min(max(1, n_valid), n_items - 1)
    permutation = np.random.default_rng(seed).permutation(n_items).tolist()
    valid = permutation[:n_valid]
    train = permutation[n_valid:]
    return SplitIndices(train=train, valid=valid)


def load_prepared_task(
    paths: DataPaths = DataPaths(),
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    seed: int = DEFAULT_SEED,
) -> PreparedTask:
    """Load the fixed ECG data and deterministic validation split."""
    missing = _missing_paths([paths.train_h5, paths.train_csv, paths.records])
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing ECG data files. Prepare the data artifacts before running autoresearch:\n"
            f"{formatted}"
        )

    try:
        import h5py
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing experiment dependency. Install requirements.txt before running autoresearch."
        ) from exc

    with h5py.File(paths.train_h5, "r") as h5_file:
        traces = np.asarray(h5_file["tracings"][()], dtype=np.float32)

    record_ids = [parse_record_id(line) for line in paths.records.read_text().splitlines() if line.strip()]
    labels_df = pd.read_csv(paths.train_csv).set_index("id_exam")
    labels_df = labels_df.reindex(record_ids)
    if labels_df["AF"].isna().any():
        raise ValueError("Could not align every RECORDS entry with an AF label in train.csv.")

    labels = labels_df["AF"].to_numpy(dtype=np.float32)
    if len(traces) != len(labels):
        raise ValueError(f"Trace count ({len(traces)}) does not match label count ({len(labels)}).")

    split = make_split_indices(len(labels), valid_fraction=valid_fraction, seed=seed)
    return PreparedTask(traces=traces, labels=labels, split=split, paths=paths)


def _binary_ranks(y_true: np.ndarray, y_score: np.ndarray) -> tuple[int, int, float]:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return len(positives), len(negatives), float("nan")

    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return len(positives), len(negatives), wins / (len(positives) * len(negatives))


def _f1_at_threshold(truth: np.ndarray, prob: np.ndarray, threshold: float) -> float:
    pred = (prob >= threshold).astype(np.float64)
    tp = float(np.sum((pred == 1) & (truth == 1)))
    fp = float(np.sum((pred == 1) & (truth == 0)))
    fn = float(np.sum((pred == 0) & (truth == 1)))
    denominator = (2 * tp) + fp + fn
    return 0.0 if denominator == 0 else float((2 * tp) / denominator)


def classification_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """Compute the fixed validation metrics used by the experiment loop."""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if truth.shape != prob.shape:
        raise ValueError("y_true and y_prob must have the same flattened shape.")
    if len(truth) == 0:
        raise ValueError("Cannot score an empty validation set.")

    _, _, auroc = _binary_ranks(truth, prob)

    order = np.argsort(-prob)
    sorted_truth = truth[order]
    positive_total = float(np.sum(sorted_truth == 1))
    if positive_total == 0:
        average_precision = float("nan")
    else:
        cumulative_positive = np.cumsum(sorted_truth == 1)
        precision_at_k = cumulative_positive / (np.arange(len(sorted_truth)) + 1)
        average_precision = float(np.sum(precision_at_k[sorted_truth == 1]) / positive_total)

    pred_at_threshold = (prob >= threshold).astype(np.float64)
    tn = float(np.sum((pred_at_threshold == 0) & (truth == 0)))
    tp = float(np.sum((pred_at_threshold == 1) & (truth == 1)))
    f1_at_threshold = _f1_at_threshold(truth, prob, threshold)

    return {
        "auroc": float(auroc),
        "average_precision": float(average_precision),
        "f1_at_0_5": f1_at_threshold,
        "accuracy": float((tp + tn) / len(truth)),
    }


def primary_metric(metrics: Dict[str, float]) -> float:
    """Return the metric used for AutoResearch keep/discard decisions."""
    return metrics["f1_at_0_5"]
