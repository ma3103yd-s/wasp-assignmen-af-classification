# ECG AF classification AutoResearch

This is an AutoResearch-style experiment to improve `autoresearch/train.py` on atrial fibrillation classification from ECG traces.

## Setup
To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date such as `jun22`. The branch `autoresearch/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the repository baseline.
3. **Read the in-scope files**:
   - `autoresearch/program.md` - this control plane.
   - `autoresearch/train.py` - the file you modify. Model architecture, optimizer, training loop, or equivalent task-specific research logic.
4. **Verify required inputs exist**: check for `codesubset/train.h5`, `codesubset/train.csv`, and `codesubset/train/RECORDS.txt`.
5. **Initialize `autoresearch/results.tsv`**: create it with just the header row if it does not exist.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation
Each experiment runs under a fixed budget of `300 seconds of training time or 15 epochs, whichever comes first`.
You launch it simply as:

`python autoresearch/train.py`

**What you CAN do:**
- Modify `autoresearch/train.py` - this is the only file you edit. Everything inside that file that belongs to the mutable research logic is fair game.

**What you CANNOT do:**
- Modify any file other than `autoresearch/train.py`.
- Change the validation split, benchmark definition, label alignment, reward function, or scoring rule.
- Add new dependencies beyond the declared environment.

**The goal is simple: get the best `f1_at_0_5` (higher is better).** The final external evaluation is F1-oriented, so F1 is the single primary keep/discard metric. Since the budget is fixed, every keep/discard decision must be based on that fixed primary metric. AUROC, average precision, validation loss, training time, and GPU/CPU usage can act as secondary diagnostics.

A valid experiment produces one set of validation probabilities per run. The fixed evaluator computes the score once. The validation set is for scoring an experiment, not for fitting decisions inside the experiment.

## Valid ML Engineering Rules

Valid changes:
- model architecture
- optimizer, learning rate, schedule, weight decay
- loss weighting or sampling based only on training labels
- training-time augmentation
- fixed inference transforms chosen before seeing validation results

Invalid changes:
- `train.py` must not inspect validation labels except through the final fixed score printed by the run.
- `train.py` must not select thresholds, ensemble members, blend weights, subsets, or post-processing choices by repeatedly scoring candidates on validation labels.
- `train.py` must not train auxiliary models on validation labels.
- `train.py` must not read record ids, filenames, row ordering, or metadata as a proxy for labels.
- `train.py` must not read data outside the fixed prepared inputs used by the data-loading path.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run the experiment as is.

## Output format
Once the script finishes it should print a grep-friendly summary such as:

```text
---
primary_metric: <fixed_validation_f1>
f1_at_0_5: <value>
auroc: <value>
average_precision: <value>
training_seconds: <value>
total_seconds: <value>
peak_resource: <0_for_cpu_or_1_for_cuda>
```

## Logging results
Log every experiment to `autoresearch/results.tsv` as tab-separated values with five columns:

```text
commit	primary_metric	resource_usage	status	description
```

Use:
- `keep` for improved results
- `discard` for non-improving results
- `crash` for failed runs

Leave `autoresearch/results.tsv` untracked by git unless the user explicitly wants otherwise.

## The experiment loop
The experiment runs on a dedicated branch such as `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at the git state: the current branch and commit you're on.
2. Tune `autoresearch/train.py` with an experimental idea by directly hacking the code.
3. `git commit`
4. Run the experiment: `python autoresearch/train.py > autoresearch/run.log 2>&1` (redirect everything - do not use tee or let output flood your context).
5. Read out the results: `grep -E 'primary_metric|f1_at_0_5|auroc|average_precision|training_seconds|peak_resource' autoresearch/run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 autoresearch/run.log` to read the stack trace and attempt a fix. If you cannot get things to work after more than a few attempts, give up.
7. Record the results in the TSV. Do not commit `autoresearch/results.tsv`; leave it untracked by git unless the user explicitly wants otherwise.
8. If the primary metric improved in the desired direction, you "advance" the branch, keeping the git commit.
9. If the primary metric is equal or worse, `git reset` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them. You are advancing the branch so that you can iterate.

## Timeout and crashes
Treat runs that exceed the intended budget by a large margin as failures unless the user has explicitly changed the rules.
Use judgment on easy-to-fix crashes, but do not let crash recovery turn into uncontrolled framework churn.

**NEVER STOP**: Once the experiment loop has begun after the initial setup, do not pause to ask the human if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". The loop runs until the human interrupts you.
