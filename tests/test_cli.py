import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ssa_estimates.cli import build_parser, main, resolve_form, resolve_quantity_names


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

    def test_top_level_help_shows_run_options(self):
        help_text = build_parser().format_help()

        self.assertIn("--quantity", help_text)
        self.assertIn("--plot-dir", help_text)
        self.assertIn("--output-format", help_text)
        self.assertIn("--run-name", help_text)
        self.assertIn("--runs-dir", help_text)
        self.assertIn("--save-samples", help_text)
        self.assertIn("--samples-dir", help_text)
        self.assertIn("gamma-min-constraint", help_text)

    def test_energy_json_smoke(self):
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
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
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertIn("quantity", stdout.getvalue())
            self.assertIn("energy", stdout.getvalue())
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["quantity"], "energy")
            self.assertEqual(data[0]["form"], "energy")
            self.assertEqual(data[0]["samples_requested"], 2)

    def test_named_run_writes_bundle(self):
        repo_root = Path(__file__).resolve().parents[1]
        fake_result = SimpleNamespace(
            mean=1.0,
            median=1.0,
            std=0.0,
            ci=(1.0, 1.0),
            samples=[1.0, 1.0],
            param_samples={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ssa_estimates.cli.mc_uncertainty", return_value=fake_result):
                with patch("ssa_estimates.cli.plot_mc_diagnostics") as plot_mock:
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        status = main(
                            [
                                "run",
                                str(repo_root / "inputs.yaml"),
                                "--quantity",
                                "energy",
                                "--samples",
                                "2",
                                "--run-name",
                                "source a",
                                "--runs-dir",
                                tmpdir,
                            ]
                        )

            run_dir = Path(tmpdir) / "source-a"
            results_file = run_dir / "results.json"
            metadata_file = run_dir / "run_metadata.json"
            input_file = run_dir / "input.yaml"

            self.assertEqual(status, 0)
            self.assertTrue(results_file.exists())
            self.assertTrue(metadata_file.exists())
            self.assertTrue(input_file.exists())
            self.assertIn("energy", stdout.getvalue())

            data = json.loads(results_file.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        self.assertEqual(data[0]["quantity"], "energy")
        self.assertEqual(metadata["run_name"], "source-a")
        self.assertEqual(metadata["form"], "energy")
        self.assertEqual(metadata["output_format"], "json")
        self.assertTrue(metadata["plots_enabled"])
        self.assertFalse(metadata["samples_saved"])
        self.assertEqual(Path(plot_mock.call_args.kwargs["save_priors_path"]).name, "priors.pdf")

    def test_save_samples_writes_dat_files(self):
        repo_root = Path(__file__).resolve().parents[1]
        fake_result = SimpleNamespace(
            mean=1.0,
            median=1.0,
            std=0.0,
            ci=(1.0, 1.0),
            samples=[1.25, 2.5],
            param_samples={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ssa_estimates.cli.mc_uncertainty", return_value=fake_result):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(
                        [
                            "run",
                            str(repo_root / "inputs.yaml"),
                            "--quantity",
                            "energy",
                            "radius",
                            "--samples",
                            "2",
                            "--save-samples",
                            "--samples-dir",
                            tmpdir,
                        ]
                    )

            energy_samples = Path(tmpdir) / "energy.dat"
            radius_samples = Path(tmpdir) / "radius.dat"

            self.assertEqual(status, 0)
            self.assertTrue(energy_samples.exists())
            self.assertTrue(radius_samples.exists())
            text = energy_samples.read_text(encoding="utf-8")

        self.assertIn("# quantity: energy", text)
        self.assertIn("# columns: sample_index value", text)
        self.assertIn("0 1.25", text)
        self.assertIn("1 2.5", text)

    def test_plot_dir_saves_one_priors_plot(self):
        repo_root = Path(__file__).resolve().parents[1]
        fake_result = SimpleNamespace(
            mean=1.0,
            median=1.0,
            std=0.0,
            ci=(1.0, 1.0),
            samples=[1.0, 1.0],
            param_samples={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("ssa_estimates.cli.mc_uncertainty", return_value=fake_result):
                with patch("ssa_estimates.cli.plot_mc_diagnostics") as plot_mock:
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = main(
                            [
                                "run",
                                str(repo_root / "inputs.yaml"),
                                "--quantity",
                                "energy",
                                "radius",
                                "--samples",
                                "2",
                                "--plot-dir",
                                tmpdir,
                            ]
                        )

        self.assertEqual(status, 0)
        self.assertEqual(plot_mock.call_count, 2)
        first_kwargs = plot_mock.call_args_list[0].kwargs
        second_kwargs = plot_mock.call_args_list[1].kwargs
        self.assertEqual(Path(first_kwargs["save_priors_path"]).name, "priors.pdf")
        self.assertIsNone(second_kwargs["save_priors_path"])


if __name__ == "__main__":
    unittest.main()
