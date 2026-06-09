import json
import tempfile
import unittest
from pathlib import Path

from ssa_estimates.cli import main, resolve_form, resolve_quantity_names


class CliTests(unittest.TestCase):
    def test_resolve_form_from_energy_keys(self):
        params = {"gamma_min": {}, "gamma_max": {}}

        self.assertEqual(resolve_form("auto", params, {}), "energy")

    def test_resolve_form_from_frequency_keys(self):
        params = {"nu_min": {}, "nu_max": {}}

        self.assertEqual(resolve_form("auto", params, {}), "frequency")

    def test_all_quantities_only_includes_supported_form(self):
        quantity_names = resolve_quantity_names(["all"], "frequency")

        self.assertIn("energy", quantity_names)
        self.assertNotIn("gamma-min-constraint", quantity_names)

    def test_energy_json_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            status = main(
                [
                    "run",
                    str(repo_root / "inputs.yaml"),
                    "--quantity",
                    "energy",
                    "--samples",
                    "2",
                    "--seed",
                    "1",
                    "--output-format",
                    "json",
                    "--output",
                    str(output),
                ]
            )

            self.assertEqual(status, 0)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["quantity"], "energy")
            self.assertEqual(data[0]["form"], "energy")
            self.assertEqual(data[0]["samples_requested"], 2)


if __name__ == "__main__":
    unittest.main()
