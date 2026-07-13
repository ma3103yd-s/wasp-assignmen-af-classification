import json
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


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

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local test environment")
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
