import unittest
from pathlib import Path


class ProgramTests(unittest.TestCase):
    def test_loop_docs_do_not_reference_external_assignment_context(self):
        forbidden_terms = (
            "leaderboard",
            "competition",
            "assignment",
            "notebook",
            "assignment_ecg",
        )
        checked_paths = [Path("autoresearch/program.md")]

        for path in checked_paths:
            content = path.read_text().lower()
            for term in forbidden_terms:
                with self.subTest(path=str(path), term=term):
                    self.assertNotIn(term, content)

    def test_readme_autoresearch_section_stays_context_neutral(self):
        content = Path("README.md").read_text().lower()
        autoresearch_section = content.split("## autoresearch loop", 1)[1]

        self.assertNotIn("leaderboard", autoresearch_section)
        self.assertNotIn("notebook", autoresearch_section)


if __name__ == "__main__":
    unittest.main()
