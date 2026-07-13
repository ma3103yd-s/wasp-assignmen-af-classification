# ECG Leaderboard Improvement Design

## Objective

Improve the raw-signal ECG classifier in `assignment_ecg_classification.ipynb` from the current `group_14_sub_1` result near F1 0.937 toward the leaderboard top-five range, which is approximately F1 0.950 in the supplied snapshot. The work will produce four carefully compared candidates for the remaining submission slots. The user will submit those candidates manually; the notebook will not contain credentials or execute leaderboard uploads.

## Fixed constraints

- The primary artifact remains `assignment_ecg_classification.ipynb`.
- The notebook remains independent of `autoresearch/`.
- The existing assignment cell order and explanatory markdown remain recognizable.
- Raw preprocessed ECG traces are the primary input with shape `(batch, 4096, 8)`.
- The HDF5 dataset is `tracings`; trace order is aligned to labels through `train/RECORDS.txt` identifiers.
- The primary split is deterministic with seed `42` and an 80/20 train/validation partition.
- Training uses `BCEWithLogitsLoss` with `pos_weight` computed from the training labels and AdamW.
- Validation reports loss, accuracy, F1 at threshold `0.5`, AUROC, and average precision.
- Each candidate checkpoint is selected by highest validation F1, with validation loss as the tie-breaker.
- Test-time prediction instantiates the same `Model` definition used for training.
- Leaderboard credentials, team ID, passwords, and submission notes remain blank for the user.
- Downloaded data, checkpoints, executed notebook outputs, and generated predictions stay outside commits.

## Model architecture

The `Model` class will accept `(batch, 4096, 8)`, transpose to channel-first `(batch, 8, 4096)`, and return one binary AF logit.

The feature extractor will use four residual multi-scale blocks. Every block has a 1x1 bottleneck, parallel temporal convolution branches with kernels `(9, 19, 39)`, a max-pooling plus 1x1-convolution branch, batch normalization, ReLU, dropout, and a squeeze-and-excitation channel gate. The block output channel widths are 128, 128, 192, and 256 respectively. A 2x temporal reduction is applied after each of the first three blocks, producing resolutions 4096, 2048, 1024, and 512 samples. Residual 1x1 shortcuts project channel counts when required.

The classifier concatenates adaptive global average pooling and adaptive global max pooling, applies dropout, and emits one logit through a linear layer. This preserves precise beat morphology in early layers while making later layers sensitive to longer rhythm context without flattening the full time axis.

No spectrogram branch, external pretrained model, demographic feature branch, or code from `autoresearch/` will be introduced.

## Training and augmentation

The fixed train/validation indices are generated once from `torch.Generator().manual_seed(42)`. All validation and test examples pass through the model unchanged.

The augmentation pipeline is enabled only for training examples and uses bounded, reproducible transforms:

- per-record amplitude scaling sampled in `[0.90, 1.10]`;
- additive Gaussian noise with scale up to `0.01` times each lead's standard deviation;
- a bounded zero-padded time shift of at most 80 samples;
- independent lead dropout with probability `0.10`.

The pipeline will not reverse ECG time, alter validation labels, or randomly crop away most of the ten-second recording. A normalization variant will apply per-record, per-lead standardization `(x - mean) / (std + 1e-6)` before the same augmentation so that normalization can be compared rather than silently replacing the raw-signal path.

The training schedule uses AdamW, a five-epoch linear warmup followed by cosine decay, gradient-norm clipping at `1.0`, mixed precision when CUDA is available, and a maximum of 120 epochs. The best checkpoint is retained even if later epochs overfit. Augmented candidates additionally maintain an exponential-moving-average parameter set with decay `0.995`; validation and checkpointing use the EMA weights.

## Candidate matrix

The existing `group_14_sub_1` consumes the first leaderboard slot. The notebook will produce four new candidate prediction sets:

1. `candidate_2_tuned`: the new multi-resolution model without augmentation, seed `42`, using the longer scheduled run. This is the low-risk architecture/training control.
2. `candidate_3_augmented_ema`: the same model with training-only augmentation and EMA weights, seed `42`.
3. `candidate_4_seed_ensemble`: the augmented, EMA model trained with explicit model seeds `42`, `43`, and `44` on the same fixed split; test logits are averaged before sigmoid.
4. `candidate_5_blended_ensemble`: a deterministic blend of the raw augmented ensemble and a per-lead-normalized augmented ensemble. The blend weight is selected from `{0.00, 0.25, 0.50, 0.75, 1.00}` by validation F1 at threshold `0.5`, with validation loss as the tie-breaker. The selected weight and all component metrics are printed and saved.

The split seed remains `42` in every candidate. The additional model seeds are explicit experiment settings used only to diversify the ensemble and are recorded with the candidate metrics. No candidate may inspect test labels or use leaderboard results during training.

Each candidate writes its own checkpoint, validation metrics, and two-column `(negative_probability, positive_probability)` prediction artifact. The strongest individual checkpoint is also copied to `model.pth` for compatibility with the assignment's standard test cell. Ensemble candidates load the same `Model` class from their component checkpoints. The selected candidate remains available through the notebook's existing `soft_pred` variable.

## Colab execution

The GPU workflow will use a named Colab session with an explicitly requested free-tier T4. Before data-dependent execution, both `colab status` and an in-kernel check of `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` must confirm the requested accelerator. A temporary notebook subset will run only dependency/import checks, dataset download, archive extraction, and CUDA verification; it will not launch training.

After the smoke test, the complete notebook will run top-to-bottom. Candidate artifacts and the executed `_output.ipynb` will be kept in a temporary or Colab output directory. If the T4 cannot be allocated or CUDA is unavailable in the kernel, the run is reported as blocked rather than silently switching accelerators or claiming a result. The named session will be stopped after execution and the session list checked for active assignments.

## Verification

Before cloud execution:

- parse the notebook as valid JSON;
- compile every code cell after removing notebook-only magics and shell commands as needed for static checking;
- verify the model definition, training setup, checkpoint loader, prediction cell, and credential fields are internally consistent;
- run `rtk python3 -m unittest discover -s tests -v`.

After cloud execution:

- confirm the smoke-test imports and CUDA checks;
- confirm HDF5 shape and `RECORDS.txt`/CSV identifier alignment;
- confirm every candidate has finite validation metrics and prediction probabilities with the expected row count and two columns whose rows sum to one;
- reload every saved checkpoint through `Model` and reproduce its candidate predictions;
- record the Colab hardware, base seed, model seeds, augmentation settings, schedule, checkpoint-selection rule, validation metrics, and artifact names.

Leaderboard scores will only be reported after the user submits the candidate artifacts and receives the actual server evaluation. No leaderboard score will be inferred from local or validation metrics.
