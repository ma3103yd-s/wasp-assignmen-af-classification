"""Main editable experiment file for AutoResearch-style ECG runs."""

from __future__ import annotations

import time
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
    weight_decay: float = 1e-4
    num_epochs: int = 15
    max_train_seconds: float = 300.0
    max_time_shift: int = 48
    noise_std: float = 0.02
    scale_std: float = 0.08
    lead_dropout: float = 0.04
    calibration_bias_min: float = -2.0
    calibration_bias_max: float = 2.0
    calibration_bias_steps: int = 81
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
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
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


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int, bottleneck_channels: int = 32):
        super().__init__()
        if in_channels > bottleneck_channels:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
                nn.GroupNorm(8, bottleneck_channels),
                nn.SiLU(),
            )
            conv_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_channels = in_channels

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(conv_channels, branch_channels, kernel_size=9, padding=4, bias=False),
                nn.Conv1d(conv_channels, branch_channels, kernel_size=19, padding=9, bias=False),
                nn.Conv1d(conv_channels, branch_channels, kernel_size=39, padding=19, bias=False),
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        out_channels = branch_channels * 4
        self.norm = nn.GroupNorm(8, out_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, traces: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(traces)
        features = [branch(reduced) for branch in self.branches]
        features.append(self.pool_branch(traces))
        merged = self.norm(torch.cat(features, dim=1))
        return self.activation(merged + self.shortcut(traces))


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
            InceptionBlock(32, 32),
            InceptionBlock(128, 32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            InceptionBlock(128, 48),
            InceptionBlock(192, 48),
            nn.MaxPool1d(kernel_size=2, stride=2),
            InceptionBlock(192, 64),
            InceptionBlock(256, 64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1),
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
        torch.backends.cudnn.benchmark = True


def make_loaders(config: TrainConfig):
    prepared = load_prepared_task(seed=DEFAULT_SEED)
    train_indices = np.asarray(prepared.split.train)
    train_traces = prepared.traces[train_indices]
    lead_mean = train_traces.mean(axis=(0, 1), keepdims=True)
    lead_std = train_traces.std(axis=(0, 1), keepdims=True)
    traces = (prepared.traces - lead_mean) / np.maximum(lead_std, 1e-6)
    traces = torch.tensor(traces, dtype=torch.float32)
    labels = torch.tensor(prepared.labels, dtype=torch.float32).reshape(-1, 1)

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
    train_labels = prepared.labels[train_indices]
    n_positive = float(train_labels.sum())
    n_negative = float(len(train_labels) - n_positive)
    pos_weight = (n_negative / max(1.0, n_positive)) ** 0.5
    return train_loader, valid_loader, prepared.traces.shape[-1], pos_weight


def augment_traces(traces: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.max_time_shift > 0:
        offset = int(
            torch.randint(
                -config.max_time_shift,
                config.max_time_shift + 1,
                size=(1,),
                device=traces.device,
            ).item()
        )
        if offset:
            traces = torch.roll(traces, shifts=offset, dims=1)

    if config.scale_std > 0:
        scale = 1.0 + (config.scale_std * torch.randn(len(traces), 1, 1, device=traces.device))
        traces = traces * scale

    if config.lead_dropout > 0:
        keep = torch.rand(len(traces), 1, traces.shape[-1], device=traces.device)
        keep = (keep > config.lead_dropout).to(traces.dtype)
        traces = traces * keep

    if config.noise_std > 0:
        traces = traces + (config.noise_std * torch.randn_like(traces))

    return traces


def train_epoch(model, dataloader, optimizer, scheduler, loss_function, device, config) -> float:
    model.train()
    total_loss = 0.0
    n_entries = 0
    for traces, labels in dataloader:
        traces = traces.to(device)
        labels = labels.to(device)
        traces = augment_traces(traces, config)

        optimizer.zero_grad(set_to_none=True)
        logits = model(traces)
        loss = loss_function(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        scheduler.step()

        batch_size = len(traces)
        total_loss += float(loss.detach().cpu()) * batch_size
        n_entries += batch_size
    return total_loss / max(1, n_entries)


def predict_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for traces, labels in dataloader:
            traces = traces.to(device)
            logits = model(traces)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits).reshape(-1)
    labels = np.concatenate(all_labels).reshape(-1)
    return labels, logits


def choose_logit_bias(labels: np.ndarray, logits: np.ndarray, config: TrainConfig) -> float:
    best_bias = 0.0
    best_metric = -1.0
    for bias in np.linspace(
        config.calibration_bias_min,
        config.calibration_bias_max,
        config.calibration_bias_steps,
    ):
        probabilities = 1.0 / (1.0 + np.exp(-(logits + bias)))
        metric = primary_metric(classification_metrics(labels, probabilities))
        if metric > best_metric or (metric == best_metric and abs(bias) < abs(best_bias)):
            best_metric = metric
            best_bias = float(bias)
    return best_bias


def evaluate_model(model, dataloader, loss_function, device, logit_bias: float):
    labels, logits = predict_logits(model, dataloader, device)
    logits = logits + logit_bias
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    metrics = classification_metrics(labels, probabilities)
    metrics["valid_loss"] = float(
        loss_function(
            torch.tensor(logits, dtype=torch.float32, device=device).reshape(-1, 1),
            torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1),
        ).detach()
    )
    metrics["logit_bias"] = logit_bias
    return metrics


def run_experiment() -> dict[str, float]:
    config = TrainConfig()
    set_seed(DEFAULT_SEED)
    start_total = time.monotonic()
    train_loader, valid_loader, n_leads, pos_weight = make_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(n_leads=n_leads).to(device)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    train_loss = float("nan")
    completed_epochs = 0
    training_start = time.monotonic()
    for epoch in range(1, config.num_epochs + 1):
        elapsed = time.monotonic() - training_start
        if elapsed >= config.max_train_seconds:
            break

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            config,
        )
        completed_epochs = epoch

    if completed_epochs == 0:
        raise RuntimeError("No training epoch completed within the fixed budget.")

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, config.model_path)

    train_labels, train_logits = predict_logits(model, train_loader, device)
    logit_bias = choose_logit_bias(train_labels, train_logits, config)
    metrics = evaluate_model(model, valid_loader, loss_function, device, logit_bias)
    metrics["train_loss"] = train_loss
    metrics["epoch"] = float(completed_epochs)
    metrics["training_seconds"] = time.monotonic() - training_start
    metrics["total_seconds"] = time.monotonic() - start_total
    metrics["primary_metric"] = primary_metric(metrics)
    metrics["peak_resource"] = 1.0 if device.type == "cuda" else 0.0
    return metrics


def print_summary(metrics: dict[str, float]) -> None:
    print("---")
    for key in (
        "primary_metric",
        "f1_at_0_5",
        "auroc",
        "average_precision",
        "accuracy",
        "valid_loss",
        "train_loss",
        "epoch",
        "logit_bias",
        "training_seconds",
        "total_seconds",
        "peak_resource",
    ):
        print(f"{key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    print_summary(run_experiment())
