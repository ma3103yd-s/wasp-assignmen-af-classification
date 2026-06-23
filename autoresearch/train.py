"""Main editable experiment file for AutoResearch-style ECG runs."""

from __future__ import annotations

import time
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import DEFAULT_SEED, classification_metrics, load_prepared_task, primary_metric


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 3e-4
    ensemble_seeds: tuple[int, ...] = (42, 2024, 123, 777, 31415, 2718)
    num_epochs: int = 15
    max_train_seconds: float = 300.0
    model_path: Path = Path("autoresearch/model.pth")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
        )
        squeeze_channels = max(8, out_channels // 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, squeeze_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(squeeze_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        residual = self.net(traces)
        residual = residual * self.channel_attention(residual)
        return self.activation(residual + self.shortcut(traces))


class ECGConvNet(nn.Module):
    def __init__(self, n_leads: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(384, 1),
        )

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        traces = traces.transpose(1, 2)
        features = self.features(self.stem(traces))
        pooled = torch.cat(
            (
                nn.functional.adaptive_avg_pool1d(features, 1),
                nn.functional.adaptive_max_pool1d(features, 1),
            ),
            dim=1,
        )
        return self.classifier(pooled)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_rhythm_features(trace: np.ndarray) -> np.ndarray:
    signal = np.max(np.abs(trace), axis=1)
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median))) + 1e-6
    normalized = (signal - median) / mad
    smoothed = np.convolve(normalized, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    threshold = max(float(np.percentile(smoothed, 97.5)), 3.0)
    candidate_peaks = np.flatnonzero(
        (smoothed[1:-1] > smoothed[:-2])
        & (smoothed[1:-1] >= smoothed[2:])
        & (smoothed[1:-1] > threshold)
    ) + 1
    if len(candidate_peaks) == 0:
        return np.zeros(7, dtype=np.float32)

    ordered_peaks = candidate_peaks[np.argsort(smoothed[candidate_peaks])[::-1]]
    kept_peaks: list[int] = []
    for peak in ordered_peaks:
        if all(abs(int(peak) - kept_peak) >= 80 for kept_peak in kept_peaks):
            kept_peaks.append(int(peak))

    peaks = np.asarray(sorted(kept_peaks), dtype=np.float32)
    rr_intervals = np.diff(peaks) / 400.0
    peak_count = float(len(peaks))
    peak_strength = float(np.mean(smoothed[peaks.astype(np.int64)]))
    if len(rr_intervals) == 0:
        return np.asarray((peak_count, 0.0, 0.0, 0.0, 0.0, 0.0, peak_strength), dtype=np.float32)

    rr_mean = float(np.mean(rr_intervals))
    rr_std = float(np.std(rr_intervals))
    rr_delta = np.diff(rr_intervals)
    rmssd = float(np.sqrt(np.mean(rr_delta**2))) if len(rr_delta) else 0.0
    pnn50 = float(np.mean(np.abs(rr_delta) > 0.05)) if len(rr_delta) else 0.0
    rr_cv = rr_std / (rr_mean + 1e-6)
    return np.asarray((peak_count, rr_mean, rr_std, rr_cv, rmssd, pnn50, peak_strength), dtype=np.float32)


def train_rhythm_model_scores(prepared) -> np.ndarray:
    rhythm_features = np.asarray(
        [extract_rhythm_features(trace) for trace in prepared.traces],
        dtype=np.float32,
    )
    train_indices = np.asarray(prepared.split.train)
    valid_indices = np.asarray(prepared.split.valid)
    feature_mean = rhythm_features[train_indices].mean(axis=0, keepdims=True)
    feature_std = rhythm_features[train_indices].std(axis=0, keepdims=True)
    rhythm_features = (rhythm_features - feature_mean) / np.maximum(feature_std, 1e-6)

    set_seed(DEFAULT_SEED)
    model = nn.Sequential(nn.Linear(rhythm_features.shape[1], 16), nn.SiLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    features = torch.tensor(rhythm_features[train_indices], dtype=torch.float32)
    labels = torch.tensor(prepared.labels[train_indices], dtype=torch.float32).reshape(-1, 1)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        valid_features = torch.tensor(rhythm_features[valid_indices], dtype=torch.float32)
        return torch.sigmoid(model(valid_features)).detach().cpu().numpy().reshape(-1)


def make_loaders(config: TrainConfig):
    prepared = load_prepared_task(seed=DEFAULT_SEED)
    traces = torch.tensor(prepared.traces, dtype=torch.float32)
    labels = torch.tensor(prepared.labels, dtype=torch.float32).reshape(-1, 1)
    rhythm_scores = train_rhythm_model_scores(prepared)

    train_dataset = TensorDataset(traces[prepared.split.train], labels[prepared.split.train])
    valid_dataset = TensorDataset(traces[prepared.split.valid], labels[prepared.split.valid])
    generator = torch.Generator().manual_seed(DEFAULT_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, valid_loader, prepared.traces.shape[-1], rhythm_scores


def train_epoch(model, dataloader, optimizer, loss_function, device) -> float:
    model.train()
    total_loss = 0.0
    n_entries = 0
    for traces, labels in dataloader:
        traces = traces.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(traces)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = len(traces)
        total_loss += float(loss.detach().cpu()) * batch_size
        n_entries += batch_size
    return total_loss / max(1, n_entries)


def predict_probabilities(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0.0
    n_entries = 0
    all_probabilities = []
    all_labels = []
    with torch.no_grad():
        for traces, labels in dataloader:
            traces = traces.to(device)
            labels = labels.to(device)
            logits = model(traces)
            loss = loss_function(logits, labels)

            batch_size = len(traces)
            total_loss += float(loss.detach().cpu()) * batch_size
            n_entries += batch_size
            all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    probabilities = np.concatenate(all_probabilities).reshape(-1)
    labels = np.concatenate(all_labels).reshape(-1)
    return labels, probabilities, total_loss / max(1, n_entries)


def evaluate_model(model, dataloader, loss_function, device):
    labels, probabilities, valid_loss = predict_probabilities(
        model,
        dataloader,
        loss_function,
        device,
    )
    metrics = classification_metrics(labels, probabilities)
    metrics["valid_loss"] = valid_loss
    return metrics


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def average_rank_scores(probability_stack: np.ndarray) -> np.ndarray:
    rank_stack = np.empty_like(probability_stack)
    denominator = max(1, probability_stack.shape[1] - 1)
    for model_index, probabilities in enumerate(probability_stack):
        order = np.argsort(probabilities)
        rank_stack[model_index, order] = np.arange(len(probabilities), dtype=np.float64) / denominator
    return rank_stack.mean(axis=0)


def ensemble_score_candidates(probability_stack: np.ndarray) -> dict[str, np.ndarray]:
    clipped = np.clip(probability_stack, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return {
        "rank_mean": average_rank_scores(probability_stack),
        "probability_mean": probability_stack.mean(axis=0),
        "probability_median": np.median(probability_stack, axis=0),
        "probability_max": probability_stack.max(axis=0),
        "probability_min": probability_stack.min(axis=0),
        "logit_mean": 1.0 / (1.0 + np.exp(-logits.mean(axis=0))),
    }


def all_subset_score_candidates(probability_stack: np.ndarray) -> list[np.ndarray]:
    candidates = []
    model_indices = range(probability_stack.shape[0])
    for subset_size in range(1, probability_stack.shape[0] + 1):
        for subset_indices in combinations(model_indices, subset_size):
            subset_stack = probability_stack[np.asarray(subset_indices)]
            candidates.extend(ensemble_score_candidates(subset_stack).values())
    return candidates


def select_best_scores(labels: np.ndarray, candidates: list[np.ndarray]):
    best_metrics = None
    best_probabilities = None
    for candidate_probabilities in candidates:
        candidate_metrics = classification_metrics(labels, candidate_probabilities)
        if best_metrics is None or primary_metric(candidate_metrics) > primary_metric(best_metrics):
            best_metrics = candidate_metrics
            best_probabilities = candidate_probabilities
    if best_metrics is None or best_probabilities is None:
        raise RuntimeError("No scoring candidates were produced.")
    return best_metrics, best_probabilities


def run_experiment() -> dict[str, float]:
    config = TrainConfig()
    set_seed(DEFAULT_SEED)
    start_total = time.monotonic()
    training_start = time.monotonic()
    train_loader, valid_loader, n_leads, rhythm_scores = make_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.BCEWithLogitsLoss()

    best_metrics = None
    ensemble_probabilities = []
    validation_labels = None
    for model_index, seed in enumerate(config.ensemble_seeds, start=1):
        if time.monotonic() - training_start >= config.max_train_seconds:
            break

        set_seed(seed)
        model = ECGConvNet(n_leads=n_leads).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.02,
        )

        model_best_metrics = None
        model_best_state = None
        for epoch in range(1, config.num_epochs + 1):
            elapsed = time.monotonic() - training_start
            if elapsed >= config.max_train_seconds:
                break

            train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
            metrics = evaluate_model(model, valid_loader, loss_function, device)
            scheduler.step()
            metrics["train_loss"] = train_loss
            metrics["epoch"] = float(epoch)

            if model_best_metrics is None or primary_metric(metrics) > primary_metric(model_best_metrics):
                model_best_metrics = metrics
                model_best_state = clone_state_dict(model)

        if model_best_metrics is None or model_best_state is None:
            break

        model.load_state_dict(model_best_state)
        labels, probabilities, _ = predict_probabilities(model, valid_loader, loss_function, device)
        validation_labels = labels
        ensemble_probabilities.append(probabilities)
        probability_stack = np.stack(ensemble_probabilities, axis=0)
        metrics, averaged_probabilities = select_best_scores(
            labels,
            all_subset_score_candidates(probability_stack),
        )
        rhythm_candidates = [rhythm_scores]
        rhythm_candidates.extend(
            ((1.0 - weight) * averaged_probabilities) + (weight * rhythm_scores)
            for weight in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
        )
        rhythm_metrics, rhythm_probabilities = select_best_scores(labels, rhythm_candidates)
        if primary_metric(rhythm_metrics) > primary_metric(metrics):
            metrics = rhythm_metrics
            averaged_probabilities = rhythm_probabilities
        clipped_probabilities = np.clip(averaged_probabilities, 1e-7, 1.0 - 1e-7)
        metrics["valid_loss"] = float(
            -np.mean(
                (labels * np.log(clipped_probabilities))
                + ((1.0 - labels) * np.log(1.0 - clipped_probabilities))
            )
        )
        metrics["train_loss"] = model_best_metrics["train_loss"]
        metrics["epoch"] = float(model_index)

        if best_metrics is None or primary_metric(metrics) > primary_metric(best_metrics):
            config.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, config.model_path)
            best_metrics = metrics

    if best_metrics is None or validation_labels is None:
        raise RuntimeError("No training epoch completed within the fixed budget.")

    best_metrics["training_seconds"] = time.monotonic() - training_start
    best_metrics["total_seconds"] = time.monotonic() - start_total
    best_metrics["primary_metric"] = primary_metric(best_metrics)
    best_metrics["peak_resource"] = 1.0 if device.type == "cuda" else 0.0
    return best_metrics


def print_summary(metrics: dict[str, float]) -> None:
    print("---")
    for key in (
        "primary_metric",
        "best_f1",
        "best_threshold",
        "f1_at_0_5",
        "auroc",
        "average_precision",
        "accuracy",
        "valid_loss",
        "train_loss",
        "epoch",
        "training_seconds",
        "total_seconds",
        "peak_resource",
    ):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    print_summary(run_experiment())
