import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoresearch.prepare import (
    DataPaths,
    classification_metrics,
    load_prepared_task,
    make_split_indices,
    primary_metric,
)


class PrepareTests(unittest.TestCase):
    def test_make_split_indices_is_deterministic_and_non_overlapping(self):
        first = make_split_indices(n_items=10, valid_fraction=0.3, seed=42)
        second = make_split_indices(n_items=10, valid_fraction=0.3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first.train), 7)
        self.assertEqual(len(first.valid), 3)
        self.assertEqual(set(first.train).intersection(first.valid), set())
        self.assertEqual(sorted(first.train + first.valid), list(range(10)))

    def test_classification_metrics_reports_fixed_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.05, 0.4, 0.35, 0.9])

        metrics = classification_metrics(y_true, y_prob)

        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["f1_at_0_5"], 2 / 3)
        self.assertNotIn("best_f1", metrics)
        self.assertNotIn("best_threshold", metrics)
        self.assertEqual(primary_metric(metrics), metrics["f1_at_0_5"])

    def test_load_prepared_task_reorders_labels_to_records_file(self):
        try:
            import h5py
            import pandas as pd
        except ModuleNotFoundError as exc:
            self.skipTest(f"experiment data dependency unavailable: {exc.name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "train.h5"
            csv_path = root / "train.csv"
            records_path = root / "RECORDS.txt"

            traces = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
            with h5py.File(h5_path, "w") as h5_file:
                h5_file.create_dataset("tracings", data=traces)

            pd.DataFrame(
                {
                    "id_exam": [3, 1, 2],
                    "AF": [1, 0, 1],
                }
            ).to_csv(csv_path, index=False)
            records_path.write_text("TNMG000002\nTNMG000001\nTNMG000003\n")

            prepared = load_prepared_task(
                DataPaths(train_h5=h5_path, train_csv=csv_path, records=records_path),
                valid_fraction=1 / 3,
                seed=7,
            )

            self.assertEqual(prepared.traces.shape, (3, 4, 2))
            self.assertEqual(prepared.labels.tolist(), [1.0, 0.0, 1.0])
            self.assertEqual(len(prepared.split.train), 2)
            self.assertEqual(len(prepared.split.valid), 1)


if __name__ == "__main__":
    unittest.main()
