# ECG Inception Notebook Design

## Objective

Complete the coding tasks in `assignment_ecg_classification.ipynb` with a self-contained, leaderboard-oriented ECG classifier that does not depend on the `autoresearch/` directory. The notebook should remain understandable as an assignment submission and runnable in Colab after the dataset is downloaded and preprocessed.

## Constraints

- Local development is CPU-only, so full training and performance validation will be deferred to Colab or another cloud/GPU setting.
- The notebook should keep the assignment's existing structure and fill the required task cells without introducing unrelated refactors.
- Leaderboard registration credentials, team ID, and submission notes remain blank for the student to fill.
- The implemented model should train from raw preprocessed ECG traces shaped `(batch, 4096, 8)`.

## Selected Approach

Use a compact InceptionTime-style 1D convolutional classifier. The model transposes each ECG to `(batch, 8, 4096)`, applies several parallel temporal convolution branches with different kernel sizes, concatenates the resulting features, and uses residual shortcuts to stabilize training. A global average pooling head maps the learned temporal features to one binary AF logit.

This approach is stronger than the provided baseline because it can learn short morphology features and longer rhythm patterns in the same block, while staying simpler than a spectrogram or multi-representation pipeline.

## Notebook Changes

- Fill the data analysis task with code for class balance, demographic summaries, a naive majority-class baseline, and one preprocessed ECG plot from `train.h5`.
- Replace the placeholder `Model` with the Inception-style ECG classifier.
- Fill the training loop with forward pass, loss computation, backpropagation, optimizer step, and weighted loss accounting.
- Fill the evaluation loop with no-gradient inference, loss accounting, probabilities, and labels.
- Fill train/validation splitting with a deterministic seeded split.
- Use `BCEWithLogitsLoss`, with `pos_weight` computed from training labels to handle class imbalance.
- Use AdamW with conservative hyperparameters suitable for a deeper CNN.
- Compute validation AUROC, average precision, accuracy, and F1 at threshold `0.5` after each epoch.
- Save `model.pth` based on the best validation F1, using validation loss as a secondary tie-breaker.
- Plot train and validation loss curves.
- Update test prediction to instantiate the final `Model`.

## Future Work

A spectrogram or time-frequency representation is reasonable future work, either as a separate 2D CNN experiment or as an auxiliary branch combined with the raw-signal model. It is not part of this first implementation because it adds preprocessing choices and runtime cost, and it may blur sharp ECG morphology that is important for AF classification.

## Verification

Because the dataset is not present locally and the machine is CPU-only, local verification will focus on:

- Notebook JSON validity.
- Static inspection of filled task cells.
- Lightweight import/syntax checks where possible.

Full verification will be done later in Colab/cloud by running the notebook top-to-bottom after downloading and preprocessing `codesubset`.
