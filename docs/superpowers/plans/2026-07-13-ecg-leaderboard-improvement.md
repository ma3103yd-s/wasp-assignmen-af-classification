# ECG Leaderboard Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `assignment_ecg_classification.ipynb` with a stronger raw-signal model, four reproducible candidate prediction paths, and enough static/cloud verification to let the user spend the remaining leaderboard submissions on measured candidates.

**Architecture:** Preserve the assignment notebook and its task order. Replace the current Inception block with a residual, squeeze-and-excitation, multi-resolution InceptionTime-style extractor; keep validation on one fixed seed-42 80/20 split; train single models and explicit seed ensembles; then reload the same `Model` class to create candidate test probabilities.

**Tech Stack:** Jupyter notebook JSON/nbformat 4, Python 3, PyTorch, NumPy, pandas, h5py, scikit-learn metrics, Matplotlib, Colab CLI with a named T4 session, and the existing `unittest` suite.

## Global Constraints

- Keep `assignment_ecg_classification.ipynb` as the primary artifact and preserve its cell order and assignment narrative.
- Do not import, copy, or modify `autoresearch/`.
- Keep the raw input contract `(batch, 4096, 8)` and the HDF5 dataset name `tracings`.
- Align labels through `codesubset/train/RECORDS.txt`, never through an assumed CSV/HDF5 row order.
- Use split seed `42`, an 80/20 train/validation split, weighted `BCEWithLogitsLoss`, and AdamW.
- Select each individual checkpoint by validation F1 at threshold `0.5`, with validation loss as the tie-breaker.
- Leave `team_id`, `password`, and `note` assignment fields empty.
- Keep datasets, checkpoints, executed notebooks, and prediction artifacts out of commits.
- Use `rtk` at the start of every shell command.
- Do not run full data-dependent training locally; the local machine is CPU-only and lacks `codesubset/`.
- Request a free-tier T4 explicitly in Colab and verify both the allocated hardware and in-kernel CUDA device before training.

## File Map

- Modify: `assignment_ecg_classification.ipynb`
  - Cell 25: model and model-supporting layers.
  - Cells 29 and 32: augmentation, EMA, training, evaluation, and metric helpers.
  - Cells 37–39: deterministic data preparation, candidate configuration, training, checkpointing, and validation-side ensemble selection.
  - Cells 42–43: test loading and candidate prediction generation through the same `Model` class.
  - Cells 36, 48, and 49: concise explanations and candidate submission table.
- Create: `tests/test_ecg_notebook_static.py`
  - Notebook JSON, model-shape, architecture-marker, candidate-contract, and credential-preservation checks.
- Create and commit: `docs/superpowers/plans/2026-07-13-ecg-leaderboard-improvement.md` (this plan).
- Runtime-only: `/tmp/ecg-colab-*` and Colab output directories for smoke notebooks, checkpoints, metrics, and predictions. These are not repository artifacts.

---

### Task 1: Add notebook regression checks before changing the implementation

**Files:**
- Create: `tests/test_ecg_notebook_static.py`
- Test: `tests/test_ecg_notebook_static.py`

**Interfaces:**
- Consumes: `assignment_ecg_classification.ipynb`.
- Produces: a local `unittest` command that fails while the candidate orchestration and new architecture are absent, then passes after Tasks 2–6.

- [ ] **Step 1: Write the failing static/model-contract test**

Create the test file with this exact content:

```python
import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "assignment_ecg_classification.ipynb"


class ECGNotebookStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
            cls.notebook = json.load(handle)
        cls.code_sources = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.all_code = "\n\n".join(cls.code_sources)

    def _model_source(self):
        return next(
            source
            for source in self.code_sources
            if "class Model(" in source and "def forward" in source
        )

    @staticmethod
    def _sanitize_notebook_commands(source):
        sanitized = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("!", "%")):
                indentation = line[: len(line) - len(stripped)]
                sanitized.append(indentation + "pass")
            else:
                sanitized.append(line)
        return "\n".join(sanitized)

    def test_notebook_has_valid_structure_and_empty_outputs(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIn("cells", self.notebook)
        self.assertGreaterEqual(len(self.notebook["cells"]), 50)
        for cell in self.notebook["cells"]:
            self.assertIn(cell["cell_type"], {"code", "markdown"})
            self.assertIsInstance(cell.get("metadata", {}), dict)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])

    def test_code_cells_compile_after_notebook_commands_are_sanitized(self):
        for index, source in enumerate(self.code_sources):
            compile(
                self._sanitize_notebook_commands(source),
                f"notebook_code_cell_{index}",
                "exec",
            )

    def test_model_preserves_input_and_output_shape_contract(self):
        namespace = {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
        }
        exec(self._model_source(), namespace)
        model = namespace["Model"]().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 4096, 8))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_model_and_candidate_markers_are_present(self):
        for marker in (
            "class SqueezeExcitation1D",
            "class TemporalDownsample1D",
            "AdaptiveMaxPool1d",
            "BCEWithLogitsLoss",
            "best_f1",
            "model.pth",
            "candidate_2_tuned",
            "candidate_3_augmented_ema",
            "candidate_4_seed_ensemble",
            "candidate_5_blended_ensemble",
        ):
            self.assertIn(marker, self.all_code)

    def test_candidate_seeds_and_fixed_split_are_explicit(self):
        self.assertRegex(self.all_code, r"seed\s*=\s*42")
        self.assertRegex(self.all_code, r"valid_fraction\s*=\s*0\.2")
        self.assertRegex(self.all_code, r"model_seeds\s*=\s*\(42,\s*43,\s*44\)")
        self.assertIn("pos_weight", self.all_code)
        self.assertIn("threshold", self.all_code)

    def test_submission_credentials_and_note_remain_blank(self):
        submission_sources = "\n".join(
            "".join(self.notebook["cells"][index].get("source", []))
            for index in (46, 47)
        )
        self.assertRegex(submission_sources, r"team_id\s*=\s*['\"]\s*['\"]")
        self.assertRegex(submission_sources, r"password\s*=\s*['\"]\s*['\"]")
        self.assertRegex(submission_sources, r"note\s*=\s*['\"]\s*['\"]")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify it fails for the intended reason**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static -v
```

Expected: the notebook structure and current model shape checks pass, while the architecture/candidate-marker test fails because the new squeeze-and-excitation/downsampling classes and candidate names do not yet exist.

- [ ] **Step 3: Check the test itself for accidental scope violations**

Run:

```bash
rtk git diff --check
```

Expected: no whitespace errors. The test must only inspect the notebook and must not require `codesubset/`, a checkpoint, network access, or leaderboard credentials.

- [ ] **Step 4: Commit the red test**

Run:

```bash
rtk git add tests/test_ecg_notebook_static.py
rtk git commit -m "test: add ECG notebook contract checks"
```

Expected: one new test file is committed; `AGENTS.md` remains untracked and is not staged.

---

### Task 2: Replace the model cell with the multi-resolution residual architecture

**Files:**
- Modify: `assignment_ecg_classification.ipynb` cell 25, the code cell beginning `class InceptionBlock1D`.
- Test: `tests/test_ecg_notebook_static.py::ECGNotebookStaticTests.test_model_preserves_input_and_output_shape_contract`.

**Interfaces:**
- Consumes: tensors shaped `(batch, 4096, 8)`.
- Produces: a `Model` class whose `forward` returns `(batch, 1)` logits, plus `SqueezeExcitation1D`, `InceptionBlock1D`, and `TemporalDownsample1D` helpers defined in the same notebook cell.

- [ ] **Step 1: Replace cell 25 through a structured nbformat edit**

Use a temporary Python script with `nbformat.read(..., as_version=4)` and `nbformat.write(...)`; do not hand-edit escaped notebook JSON. Set cell 25's source to the following implementation:

```python
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(9, 19, 39),
        bottleneck_channels=32,
        dropout=0.10,
    ):
        super().__init__()
        self.bottleneck = nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        merged_channels = out_channels * (len(kernel_sizes) + 1)
        self.norm = nn.BatchNorm1d(merged_channels)
        self.activation = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation1D(merged_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, merged_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(merged_channels),
            )
            if in_channels != merged_channels
            else nn.Identity()
        )

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(x))
        merged = self.activation(self.norm(torch.cat(branches, dim=1)))
        merged = self.dropout(self.se(merged))
        return self.activation(merged + self.shortcut(x))


class TemporalDownsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            InceptionBlock1D(8, 32, bottleneck_channels=16, dropout=0.08),
            TemporalDownsample1D(128),
            InceptionBlock1D(128, 32, bottleneck_channels=32, dropout=0.10),
            TemporalDownsample1D(128),
            InceptionBlock1D(128, 48, bottleneck_channels=32, dropout=0.12),
            TemporalDownsample1D(192),
            InceptionBlock1D(192, 64, bottleneck_channels=48, dropout=0.15),
        )
        self.average_pool = nn.AdaptiveAvgPool1d(1)
        self.maximum_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        pooled = torch.cat(
            [self.average_pool(x).flatten(1), self.maximum_pool(x).flatten(1)],
            dim=1,
        )
        return self.classifier(pooled)
```

- [ ] **Step 2: Run the focused model contract test**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static.ECGNotebookStaticTests.test_model_preserves_input_and_output_shape_contract -v
```

Expected: PASS with output shape `(2, 1)` and no `codesubset/` access.

- [ ] **Step 3: Verify the architecture source has no accidental dependency**

Run:

```bash
rtk rg -n "autoresearch|spectrogram|team_id|password|note" assignment_ecg_classification.ipynb
```

Expected: no new model-cell dependency on `autoresearch/`, no spectrogram code, and the only credential matches remain in the original blank submission cells.

- [ ] **Step 4: Commit the model change**

Run:

```bash
rtk git add assignment_ecg_classification.ipynb
rtk git commit -m "feat: strengthen ECG inception model"
```

Expected: only the notebook model cell is included in this commit.

---

### Task 3: Add deterministic augmentation, EMA, training, and validation helpers

**Files:**
- Modify: `assignment_ecg_classification.ipynb` cell 29, the training-loop cell.
- Modify: `assignment_ecg_classification.ipynb` cell 32, the evaluation-loop cell.
- Test: `tests/test_ecg_notebook_static.py` model and marker checks.

**Interfaces:**
- Consumes: `(batch, 4096, 8)` tensors, binary labels shaped `(batch, 1)`, an AdamW optimizer, a weighted BCE loss, and a CUDA-or-CPU device.
- Produces: `augment_ecg_batch`, `ModelEMA`, `train_loop(..., scaler, augment=False, ema=None)`, `eval_loop(...)`, and `compute_metrics(...)`.

- [ ] **Step 1: Add the training-only ECG augmentation and EMA helpers to cell 29**

Set cell 29's source to begin with these helpers, followed by the training loop:

```python
def augment_ecg_batch(traces):
    augmented = traces.clone()
    batch_size = augmented.shape[0]

    scale = torch.empty((batch_size, 1, 1), device=augmented.device).uniform_(0.90, 1.10)
    augmented = augmented * scale

    lead_std = augmented.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise_scale = torch.empty((batch_size, 1, 1), device=augmented.device).uniform_(0.0, 0.01)
    augmented = augmented + torch.randn_like(augmented) * lead_std * noise_scale

    shifts = torch.randint(-80, 81, (batch_size,), device=augmented.device)
    for index, shift in enumerate(shifts.tolist()):
        if shift == 0:
            continue
        shifted = torch.roll(augmented[index], shifts=shift, dims=0)
        if shift > 0:
            shifted[:shift] = 0
        else:
            shifted[shift:] = 0
        augmented[index] = shifted

    dropped_leads = torch.rand((batch_size, augmented.shape[2]), device=augmented.device) < 0.10
    augmented = augmented.masked_fill(dropped_leads.unsqueeze(1), 0.0)
    return augmented


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if self.shadow[name].is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return {name: value.detach().clone() for name, value in self.shadow.items()}


def clone_state_dict(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def train_loop(
    epoch,
    dataloader,
    model,
    optimizer,
    loss_function,
    device,
    scaler,
    augment=False,
    ema=None,
):
    model.train()
    total_loss = 0.0
    n_entries = 0
    amp_enabled = device.type == "cuda"
    train_pbar = tqdm(dataloader, desc=f"Training Epoch {epoch:2d}", leave=True)

    for traces, diagnoses in train_pbar:
        traces = traces.to(device, non_blocking=amp_enabled)
        diagnoses = diagnoses.to(device, non_blocking=amp_enabled)
        if augment:
            traces = augment_ecg_batch(traces)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(traces)
            loss = loss_function(logits, diagnoses)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = len(traces)
        total_loss += loss.detach().item() * batch_size
        n_entries += batch_size
        train_pbar.set_postfix(loss=total_loss / n_entries)

    train_pbar.close()
    return total_loss / n_entries
```

- [ ] **Step 2: Replace cell 32 with logits-preserving evaluation and fixed-threshold metrics**

Use this implementation so ensembles average logits instead of already-thresholded labels:

```python
def eval_loop(epoch, dataloader, model, loss_function, device):
    model.eval()
    total_loss = 0.0
    n_entries = 0
    valid_logits = []
    valid_true = []
    eval_pbar = tqdm(dataloader, desc=f"Evaluation Epoch {epoch:2d}", leave=True)

    with torch.no_grad():
        for traces_cpu, diagnoses_cpu in eval_pbar:
            traces = traces_cpu.to(device, non_blocking=device.type == "cuda")
            diagnoses = diagnoses_cpu.to(device, non_blocking=device.type == "cuda")
            logits = model(traces)
            loss = loss_function(logits, diagnoses)
            valid_logits.append(logits.detach().cpu().numpy())
            valid_true.append(diagnoses.detach().cpu().numpy())

            batch_size = len(traces)
            total_loss += loss.detach().item() * batch_size
            n_entries += batch_size
            eval_pbar.set_postfix(loss=total_loss / n_entries)

    eval_pbar.close()
    return (
        total_loss / n_entries,
        np.concatenate(valid_logits, axis=0),
        np.concatenate(valid_true, axis=0),
    )


def compute_metrics(logits, targets, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
    target = np.asarray(targets).reshape(-1).astype(int)
    predicted = (scores >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(target, predicted)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(target, scores)),
    }
    metrics["auroc"] = (
        float(roc_auc_score(target, scores))
        if np.unique(target).size == 2
        else float("nan")
    )
    return metrics
```

- [ ] **Step 3: Run the static test to verify the helper markers and model contract**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static -v
```

Expected: the model/candidate marker test may still fail because candidate orchestration has not been added; the model shape test and helper syntax must pass.

- [ ] **Step 4: Commit the helper change**

Run:

```bash
rtk git add assignment_ecg_classification.ipynb
rtk git commit -m "feat: add ECG augmentation and EMA training helpers"
```

Expected: only cells 29 and 32 are included in this commit.

---

### Task 4: Make data preparation and candidate configuration deterministic

**Files:**
- Modify: `assignment_ecg_classification.ipynb` cell 37, seed and hyperparameter setup.
- Modify: `assignment_ecg_classification.ipynb` cell 38, data loading and fixed split.
- Test: `tests/test_ecg_notebook_static.py::ECGNotebookStaticTests.test_candidate_seeds_and_fixed_split_are_explicit`.

**Interfaces:**
- Consumes: `codesubset/train.h5`, `codesubset/train.csv`, and `codesubset/train/RECORDS.txt`.
- Produces: `traces_raw`, `labels`, `train_indices`, `valid_indices`, `make_trace_variant(normalize)`, `make_dataloaders(normalize, loader_seed)`, `set_seed(value)`, and candidate constants.

- [ ] **Step 1: Replace cell 37 with explicit seeds, schedule, paths, and output directory**

Use this setup before constructing loaders:

```python
import math
import random
import shutil
from pathlib import Path

from torch.utils.data import DataLoader, Subset, TensorDataset


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
model_seeds = (42, 43, 44)
valid_fraction = 0.20
learning_rate = 1e-3
weight_decay = 1e-4
warmup_epochs = 5
num_epochs = 120
batch_size = 64
ema_decay = 0.995
output_dir = Path("ecg_candidate_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm.write(f"Use device: {device}")
```

- [ ] **Step 2: Replace cell 38 with aligned data loading, one fixed split, and reusable loaders**

Use `RECORDS.txt` identifiers to reindex labels and keep normalization candidate-local:

```python
path_to_h5_train = "codesubset/train.h5"
path_to_csv_train = "codesubset/train.csv"
path_to_records = "codesubset/train/RECORDS.txt"

with h5py.File(path_to_h5_train, "r") as h5_file:
    traces_raw = torch.tensor(h5_file["tracings"][()], dtype=torch.float32)

record_ids = pd.read_csv(path_to_records, header=None)[0].astype(str)
ids_traces = record_ids.str.split("TNMG", expand=True)[1].astype(int).to_numpy()
labels_frame = pd.read_csv(path_to_csv_train).set_index("id_exam").reindex(ids_traces)
labels = torch.tensor(labels_frame["AF"].to_numpy(), dtype=torch.float32).reshape(-1, 1)

if len(traces_raw) != len(labels) or not torch.isfinite(traces_raw).all():
    raise ValueError("Aligned ECG traces and labels must have equal length and finite traces")
if torch.unique(labels).numel() != 2:
    raise ValueError("The training split requires both AF classes")

base_dataset = TensorDataset(traces_raw, labels)
n_valid = int(round(len(base_dataset) * valid_fraction))
n_train = len(base_dataset) - n_valid
split_generator = torch.Generator().manual_seed(seed)
dataset_train, dataset_valid = torch.utils.data.random_split(
    base_dataset,
    [n_train, n_valid],
    generator=split_generator,
)
train_indices = list(dataset_train.indices)
valid_indices = list(dataset_valid.indices)


def make_trace_variant(normalize):
    if not normalize:
        return traces_raw
    mean = traces_raw.mean(dim=1, keepdim=True)
    std = traces_raw.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (traces_raw - mean) / std


def make_dataloaders(normalize, loader_seed):
    features = make_trace_variant(normalize)
    dataset = TensorDataset(features, labels)
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)
    generator = torch.Generator().manual_seed(loader_seed)
    common = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
        "num_workers": 0,
    }
    return (
        DataLoader(train_subset, shuffle=True, generator=generator, **common),
        DataLoader(valid_subset, shuffle=False, **common),
    )
```

- [ ] **Step 3: Run the seed/split static test**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static.ECGNotebookStaticTests.test_candidate_seeds_and_fixed_split_are_explicit -v
```

Expected: PASS and no attempt to access `codesubset/` during the static test.

- [ ] **Step 4: Commit deterministic data preparation**

Run:

```bash
rtk git add assignment_ecg_classification.ipynb
rtk git commit -m "feat: make ECG candidate data preparation reproducible"
```

Expected: only cells 37 and 38 are included in this commit.

---

### Task 5: Train candidate checkpoints and build validation-selected ensembles

**Files:**
- Modify: `assignment_ecg_classification.ipynb` cell 39, the train/validate orchestration cell.
- Test: `tests/test_ecg_notebook_static.py::ECGNotebookStaticTests.test_model_and_candidate_markers_are_present`.

**Interfaces:**
- Consumes: `Model`, `ModelEMA`, `train_loop`, `eval_loop`, `compute_metrics`, `make_dataloaders`, `train_indices`, `valid_indices`, and the explicit constants from Task 4.
- Produces: `candidate_2_tuned`, `candidate_3_augmented_ema`, `candidate_4_seed_ensemble`, `candidate_5_blended_ensemble`, `candidate_results`, `selected_candidate`, and per-candidate checkpoint/metric metadata.

- [ ] **Step 1: Add the learning-rate, checkpoint, and prediction utility functions to cell 39**

Use these exact interfaces and selection rule:

```python
def set_epoch_learning_rate(optimizer, epoch):
    if epoch <= warmup_epochs:
        factor = epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * factor


def save_checkpoint(path, state, candidate_name, model_seed, normalize, augment, epoch, metrics):
    cpu_state = {name: value.detach().cpu() for name, value in state.items()}
    torch.save(
        {
            "model": cpu_state,
            "candidate": candidate_name,
            "model_seed": model_seed,
            "normalize": normalize,
            "augment": augment,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def save_prediction_artifact(candidate_name, logits, metrics):
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
    soft_predictions = np.column_stack((1.0 - probabilities, probabilities))
    np.save(output_dir / f"{candidate_name}_soft_pred.npy", soft_predictions)
    pd.DataFrame(metrics, index=[0]).to_csv(
        output_dir / f"{candidate_name}_metrics.csv",
        index=False,
    )
    return soft_predictions


def restore_state(model, state):
    model.load_state_dict(state, strict=True)


def weighted_validation_loss(logits, targets):
    train_target = labels[train_indices].reshape(-1)
    positive = train_target.sum()
    negative = len(train_target) - positive
    pos_weight = torch.as_tensor(negative / positive, dtype=torch.float32)
    logits_tensor = torch.tensor(np.asarray(logits).reshape(-1), dtype=torch.float32)
    targets_tensor = torch.tensor(np.asarray(targets).reshape(-1), dtype=torch.float32)
    return float(
        nn.functional.binary_cross_entropy_with_logits(
            logits_tensor,
            targets_tensor,
            pos_weight=pos_weight,
        )
    )


def train_candidate(candidate_name, model_seed, normalize=False, augment=False, use_ema=False):
    set_seed(model_seed)
    train_loader, valid_loader = make_dataloaders(normalize, model_seed)
    model = Model().to(device)
    train_labels = labels[train_indices]
    num_positive = train_labels.sum()
    num_negative = len(train_labels) - num_positive
    pos_weight = (num_negative / num_positive).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    checkpoint_path = output_dir / f"{candidate_name}_model.pth"
    best_f1 = -np.inf
    best_loss = np.inf
    best_epoch = None
    best_metrics = None
    history = []

    for epoch in range(1, num_epochs + 1):
        set_epoch_learning_rate(optimizer, epoch)
        train_loss = train_loop(
            epoch,
            train_loader,
            model,
            optimizer,
            loss_function,
            device,
            scaler,
            augment=augment,
            ema=ema,
        )
        raw_state = clone_state_dict(model)
        evaluation_state = ema.state_dict() if ema is not None else raw_state
        restore_state(model, evaluation_state)
        valid_loss, valid_logits, valid_true = eval_loop(
            epoch,
            valid_loader,
            model,
            loss_function,
            device,
        )
        restore_state(model, raw_state)

        metrics = compute_metrics(valid_logits, valid_true, threshold=0.5)
        metrics.update({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        history.append(metrics)
        is_better = (
            metrics["f1"] > best_f1
            or (
                np.isclose(metrics["f1"], best_f1)
                and metrics["valid_loss"] < best_loss
            )
        )
        if is_better:
            best_f1 = metrics["f1"]
            best_loss = metrics["valid_loss"]
            best_epoch = epoch
            best_metrics = dict(metrics)
            save_checkpoint(
                checkpoint_path,
                evaluation_state,
                candidate_name,
                model_seed,
                normalize,
                augment,
                epoch,
                best_metrics,
            )

        tqdm.write(
            f"{candidate_name} epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
            f"f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f} "
            f"ap={metrics['average_precision']:.4f}"
        )

    pd.DataFrame(history).to_csv(
        output_dir / f"{candidate_name}_history.csv",
        index=False,
    )
    return {
        "name": candidate_name,
        "kind": "single",
        "checkpoint_paths": [checkpoint_path],
        "normalize": normalize,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
    }
```

- [ ] **Step 2: Add validation-logit ensemble and blend functions**

Append these functions to cell 39. They reload `Model` for every component, average logits, and select the blend weight only from the fixed validation split:

```python
def logits_for_indices(checkpoint_path, indices, normalize):
    features = make_trace_variant(normalize)
    loader = DataLoader(
        Subset(TensorDataset(features), indices),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    logits = []
    with torch.no_grad():
        for (traces_batch,) in loader:
            logits.append(model(traces_batch.to(device)).cpu().numpy())
    return np.concatenate(logits, axis=0).reshape(-1)


def build_ensemble(candidate_name, runs, normalize):
    validation_logits = np.mean(
        [
            logits_for_indices(run["checkpoint_paths"][0], valid_indices, normalize)
            for run in runs
        ],
        axis=0,
    )
    metrics = compute_metrics(
        validation_logits,
        labels[valid_indices].numpy(),
        threshold=0.5,
    )
    metrics.update(
        {
            "epoch": -1,
            "valid_loss": weighted_validation_loss(
                validation_logits,
                labels[valid_indices].numpy(),
            ),
        }
    )
    return {
        "name": candidate_name,
        "kind": "ensemble",
        "runs": runs,
        "checkpoint_paths": [run["checkpoint_paths"][0] for run in runs],
        "normalize": normalize,
        "validation_logits": validation_logits,
        "metrics": metrics,
    }


def build_blended_candidate(raw_ensemble, normalized_ensemble):
    best_weight = None
    best_metrics = None
    for weight in (0.00, 0.25, 0.50, 0.75, 1.00):
        logits = (
            (1.0 - weight) * raw_ensemble["validation_logits"]
            + weight * normalized_ensemble["validation_logits"]
        )
        metrics = compute_metrics(
            logits,
            labels[valid_indices].numpy(),
            threshold=0.5,
        )
        metrics["blend_weight"] = weight
        metrics["valid_loss"] = weighted_validation_loss(
            logits,
            labels[valid_indices].numpy(),
        )
        if best_metrics is None or (
            metrics["f1"] > best_metrics["f1"]
            or (
                np.isclose(metrics["f1"], best_metrics["f1"])
                and metrics["valid_loss"] < best_metrics["valid_loss"]
            )
        ):
            best_weight = weight
            best_metrics = metrics

    return {
        "name": "candidate_5_blended_ensemble",
        "kind": "blend",
        "raw_ensemble": raw_ensemble,
        "normalized_ensemble": normalized_ensemble,
        "blend_weight": best_weight,
        "metrics": best_metrics,
    }
```

- [ ] **Step 3: Execute the four candidate configurations in cell 39 and select the validation winner**

Append this orchestration block:

```python
candidate_2_tuned = train_candidate(
    "candidate_2_tuned",
    model_seed=42,
    normalize=False,
    augment=False,
    use_ema=False,
)

raw_augmented_runs = [
    train_candidate(
        f"raw_augmented_seed_{model_seed}",
        model_seed=model_seed,
        normalize=False,
        augment=True,
        use_ema=True,
    )
    for model_seed in model_seeds
]
candidate_3_augmented_ema = raw_augmented_runs[0]
candidate_4_seed_ensemble = build_ensemble(
    "candidate_4_seed_ensemble",
    raw_augmented_runs,
    normalize=False,
)

normalized_augmented_runs = [
    train_candidate(
        f"normalized_augmented_seed_{model_seed}",
        model_seed=model_seed,
        normalize=True,
        augment=True,
        use_ema=True,
    )
    for model_seed in model_seeds
]
normalized_ensemble = build_ensemble(
    "normalized_augmented_ensemble",
    normalized_augmented_runs,
    normalize=True,
)
candidate_5_blended_ensemble = build_blended_candidate(
    candidate_4_seed_ensemble,
    normalized_ensemble,
)

candidate_results = [
    candidate_2_tuned,
    candidate_3_augmented_ema,
    candidate_4_seed_ensemble,
    candidate_5_blended_ensemble,
]
selected_candidate = max(
    candidate_results,
    key=lambda candidate: (
        candidate["metrics"]["f1"],
        -candidate["metrics"].get("valid_loss", 0.0),
    ),
)

all_individual_runs = [candidate_2_tuned] + raw_augmented_runs + normalized_augmented_runs
best_individual = max(
    all_individual_runs,
    key=lambda candidate: (
        candidate["metrics"]["f1"],
        -candidate["metrics"]["valid_loss"],
    ),
)
shutil.copyfile(best_individual["checkpoint_paths"][0], "model.pth")
pd.DataFrame(
    [
        {"candidate": candidate["name"], **candidate["metrics"]}
        for candidate in candidate_results
    ]
).to_csv(output_dir / "candidate_summary.csv", index=False)

print(
    "Selected candidate:",
    selected_candidate["name"],
    "validation metrics:",
    selected_candidate["metrics"],
)
```

- [ ] **Step 4: Run the static candidate-marker test**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static.ECGNotebookStaticTests.test_model_and_candidate_markers_are_present -v
```

Expected: PASS, including all four candidate names, `BCEWithLogitsLoss`, `best_f1`, and `model.pth`.

- [ ] **Step 5: Commit candidate orchestration**

Run:

```bash
rtk git add assignment_ecg_classification.ipynb
rtk git commit -m "feat: train reproducible ECG candidate ensembles"
```

Expected: only the notebook training/configuration cells are included; generated `ecg_candidate_outputs/` is not staged.

---

### Task 6: Update test prediction and assignment explanations without populating credentials

**Files:**
- Modify: `assignment_ecg_classification.ipynb` cells 42–43, test loading and prediction.
- Modify: `assignment_ecg_classification.ipynb` cells 46–47, guard registration/submission calls when credentials are blank.
- Modify: `assignment_ecg_classification.ipynb` cell 36, hyperparameter explanation.
- Modify: `assignment_ecg_classification.ipynb` cell 48, submission table/explanation.
- Modify: `assignment_ecg_classification.ipynb` cell 49, metrics reflection to mention ensemble probabilities.
- Test: `tests/test_ecg_notebook_static.py::ECGNotebookStaticTests.test_submission_credentials_and_note_remain_blank`.

**Interfaces:**
- Consumes: `selected_candidate`, its checkpoint paths, `test.h5`, and `Model`.
- Produces: candidate-specific `.npy`/`.csv` probability artifacts and the assignment-compatible `soft_pred` array with shape `(number_of_test_records, 2)`.

- [ ] **Step 1: Keep cell 42 focused on aligned test loading**

Use this source for cell 42:

```python
from torch.utils.data import DataLoader, TensorDataset

path_to_h5_test = "codesubset/test.h5"
with h5py.File(path_to_h5_test, "r") as h5_file:
    test_traces_raw = torch.tensor(h5_file["tracings"][()], dtype=torch.float32)

test_dataloader = DataLoader(
    TensorDataset(test_traces_raw),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=device.type == "cuda",
    num_workers=0,
)
```

- [ ] **Step 2: Replace cell 43 with same-Model single/ensemble prediction**

The cell must instantiate `Model` for every checkpoint, load the saved `model` state, average logits for ensemble/blend candidates, and then construct the standard two-column probabilities:

```python
def test_logits_for_candidate(candidate):
    def logits_for_checkpoint(checkpoint_path, normalize):
        if normalize:
            test_mean = test_traces_raw.mean(dim=1, keepdim=True)
            test_std = test_traces_raw.std(dim=1, keepdim=True).clamp_min(1e-6)
            features = (test_traces_raw - test_mean) / test_std
        else:
            features = test_traces_raw
        loader = DataLoader(
            TensorDataset(features),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
            num_workers=0,
        )
        model = Model().to(device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        logits = []
        with torch.no_grad():
            for (traces_batch,) in loader:
                logits.append(model(traces_batch.to(device)).cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1)

    if candidate["kind"] == "single":
        return logits_for_checkpoint(
            candidate["checkpoint_paths"][0],
            candidate["normalize"],
        )
    if candidate["kind"] == "ensemble":
        return np.mean(
            [
                logits_for_checkpoint(path, candidate["normalize"])
                for path in candidate["checkpoint_paths"]
            ],
            axis=0,
        )

    raw_logits = test_logits_for_candidate(candidate["raw_ensemble"])
    normalized_logits = test_logits_for_candidate(candidate["normalized_ensemble"])
    return (
        (1.0 - candidate["blend_weight"]) * raw_logits
        + candidate["blend_weight"] * normalized_logits
    )


candidate_predictions = {}
for candidate in candidate_results:
    logits = test_logits_for_candidate(candidate)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    candidate_soft_pred = np.column_stack((1.0 - probabilities, probabilities))
    candidate_predictions[candidate["name"]] = candidate_soft_pred
    save_prediction_artifact(candidate["name"], logits, candidate["metrics"])
    pd.DataFrame(candidate_soft_pred, columns=["negative", "positive"]).to_csv(
        output_dir / f"{candidate['name']}_soft_pred.csv",
        index=False,
    )

soft_pred = candidate_predictions[selected_candidate["name"]]
print(
    "Selected prediction:",
    selected_candidate["name"],
    "shape:",
    soft_pred.shape,
)
```

- [ ] **Step 3: Update the explanation cells with the actual candidate progression**

Cell 36 should explain seed-42 split preservation, AdamW, warmup/cosine schedule, weighted BCE, augmentation-only-on-training, EMA, and validation-F1 checkpointing. Cell 48 should show the existing baseline plus the four notes `group_N_sub_2` through `group_N_sub_5`, with descriptions matching `candidate_2_tuned`, `candidate_3_augmented_ema`, `candidate_4_seed_ensemble`, and `candidate_5_blended_ensemble`. Leave the team ID and all submission-cell values blank. Cell 49 should retain the accuracy/F1/AUROC/AP definitions and mention that ensemble logits are converted to class probabilities only after averaging.

Use these guards in cells 46 and 47 so a top-to-bottom training run never contacts the leaderboard with empty values:

```python
# Cell 46
team_id = ""
password = ""

if team_id and password:
    r = register_team(team_id, password)
    if r.status_code == 201:
        print("Team registered successfully! Good luck")
    elif r.status_code != 200:
        raise Exception("The team registration failed")
else:
    print("Leaderboard registration skipped; fill team_id and password before submitting.")
```

```python
# Cell 47
note = ""

if team_id and password and note:
    r = submit(team_id, password, soft_pred.tolist(), note)
    if r.status_code == 201:
        print("Submission successful!")
    elif r.status_code == 200:
        print("Submission updated!")
else:
    print("Leaderboard submission skipped; fill credentials and note before submitting.")
```

Keep the original registration/submission instructions in markdown so the user can fill the fields manually later. When writing the edited notebook, preserve `nbformat`/`nbformat_minor`, set edited code-cell `execution_count` to `None`, and keep all source outputs empty.

- [ ] **Step 4: Run the credential and notebook-contract tests**

Run:

```bash
rtk python3 -m unittest tests.test_ecg_notebook_static -v
```

Expected: all static tests pass, including the blank `team_id`, `password`, and `note` assertions.

- [ ] **Step 5: Commit prediction and explanation changes**

Run:

```bash
rtk git add assignment_ecg_classification.ipynb tests/test_ecg_notebook_static.py
rtk git commit -m "docs: document ECG candidate submissions"
```

Expected: no dataset, checkpoint, output notebook, or prediction file is included in the commit.

---

### Task 7: Run local structural verification, Colab smoke checks, and full GPU training

**Files:**
- Verify: `assignment_ecg_classification.ipynb`.
- Verify: `tests/test_ecg_notebook_static.py` and the existing `tests/` suite.
- Runtime-only: `/tmp/ecg-colab-state.json`, `/tmp/ecg-colab-smoke.ipynb`, and Colab output directories.

**Interfaces:**
- Consumes: the edited notebook and the available Colab CLI credentials.
- Produces: verified candidate metrics/predictions in Colab, a hardware/seed/configuration record, and no committed runtime artifacts.

- [ ] **Step 1: Run the complete local static and existing test suites**

Run each command separately:

```bash
rtk git diff --check
rtk python3 -m unittest tests.test_ecg_notebook_static -v
rtk python3 -m unittest discover -s tests -v
```

Expected: all tests pass; `git diff --check` is empty. The output must explicitly note that full training is deferred because `codesubset/` is absent locally.

- [ ] **Step 2: Parse and compile notebook code cells without executing data-dependent cells**

Run:

```bash
rtk python3 -c "import ast, json; d=json.load(open('assignment_ecg_classification.ipynb')); sources=[]; [sources.append('\\n'.join((line[:len(line)-len(line.lstrip())] + 'pass') if line.lstrip().startswith(('%', '!')) else line for line in ''.join(c.get('source', [])).splitlines())) for c in d['cells'] if c.get('cell_type') == 'code']; [ast.parse(source) for source in sources]; print('notebook JSON and sanitized code cells parse')"
```

Expected: the command exits 0 and prints `notebook JSON and sanitized code cells parse`. Notebook magic/shell lines are replaced with syntactic `pass` statements for parsing; their real execution is checked only in Colab.

- [ ] **Step 3: Create the early-cell Colab smoke notebook**

Use `nbformat` in a temporary script to copy only notebook cells 4, 5, 7, and 8, then append this CUDA check cell:

```python
import torch
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Requested Colab GPU is not available inside the kernel")
print("cuda_device=", torch.cuda.get_device_name(0))
```

Write the result to `/tmp/ecg-colab-smoke.ipynb`; do not write the smoke notebook into the repository.

- [ ] **Step 4: Request and verify a named free-tier T4 session**

Run:

```bash
rtk colab --config /tmp/ecg-colab-state.json --auth=adc new -s ecg-inception-t4 --gpu T4
rtk colab --config /tmp/ecg-colab-state.json --auth=adc status -s ecg-inception-t4
```

Expected: status reports a live `ecg-inception-t4` session with T4 hardware. If allocation fails or reports another accelerator, stop the session if necessary and report the unavailability instead of launching full training.

- [ ] **Step 5: Run the smoke notebook and inspect its output**

Run:

```bash
rtk colab --config /tmp/ecg-colab-state.json --auth=adc exec -s ecg-inception-t4 -f /tmp/ecg-colab-smoke.ipynb
```

Expected: dependency/import checks complete, `codesubset.tar.gz` downloads, `codesubset/` extracts, and the CUDA cell prints `cuda_available=True` plus a T4 device name. No model training cell is present in this smoke notebook.

- [ ] **Step 6: Run the full notebook on the verified session**

Copy the edited notebook to a temporary execution path and run it:

```bash
rtk cp assignment_ecg_classification.ipynb /tmp/ecg-colab-full.ipynb
rtk colab --config /tmp/ecg-colab-state.json --auth=adc exec -s ecg-inception-t4 -f /tmp/ecg-colab-full.ipynb
```

Expected: preprocessing generates `codesubset/train.h5` and `codesubset/test.h5`; all seven model runs finish or leave a clearly identified interrupted candidate; `candidate_summary.csv` contains validation metrics; `soft_pred` artifacts have the test row count and two columns; and no leaderboard upload cell is executed because credentials/notes remain blank.

- [ ] **Step 7: Verify Colab artifacts and stop the session**

Check the output logs and candidate files for finite metrics, finite probabilities, expected row counts, row sums of one, checkpoint reloadability, recorded seeds, and selected-candidate identity. Then run:

```bash
rtk colab --config /tmp/ecg-colab-state.json --auth=adc stop -s ecg-inception-t4
rtk colab --config /tmp/ecg-colab-state.json --auth=adc sessions
```

Expected: the named session is stopped and no active session remains. Record the actual Colab hardware, validation metrics, candidate artifact names, and the exact candidate the user should submit first. Do not call any leaderboard upload endpoint.

- [ ] **Step 8: Verify the final repository state before handoff**

Run:

```bash
rtk git status --short --branch
rtk git log -6 --oneline --decorate
```

Expected: only intentional source/spec/test changes are tracked; runtime files remain outside the repository. Report measured validation results and clearly label leaderboard results as unavailable until the user submits and receives the server evaluation.
