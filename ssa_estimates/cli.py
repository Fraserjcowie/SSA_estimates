"""Command-line interface for SSA estimates."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from . import __version__, ssa_tools
from .mc_tools import mc_uncertainty, plot_mc_diagnostics


@dataclass(frozen=True)
class Quantity:
    key: str
    label: str
    energy_function: Optional[Callable[..., float]]
    frequency_function: Optional[Callable[..., float]]


@dataclass(frozen=True)
class RunBundle:
    name: str
    directory: Path
    input_path: Path
    metadata_path: Path
    results_path: Path
    plots_dir: Optional[Path]


QUANTITIES: Dict[str, Quantity] = {
    "energy": Quantity(
        "energy",
        "Minimum energy",
        ssa_tools.E_energy_form,
        ssa_tools.E_frequency_form,
    ),
    "radius": Quantity(
        "radius",
        "Source radius",
        ssa_tools.R_energy_form,
        ssa_tools.R_frequency_form,
    ),
    "magnetic-field": Quantity(
        "magnetic-field",
        "Magnetic field",
        ssa_tools.B_energy_form,
        ssa_tools.B_frequency_form,
    ),
    "n0": Quantity(
        "n0",
        "Electron distribution normalisation",
        ssa_tools.N0_energy_form,
        ssa_tools.N0_frequency_form,
    ),
    "electron-density": Quantity(
        "electron-density",
        "Electron number density",
        ssa_tools.ne_energy_form,
        ssa_tools.ne_frequency_form,
    ),
    "electron-number": Quantity(
        "electron-number",
        "Total electron number",
        ssa_tools.Ne_energy_form,
        ssa_tools.Ne_frequency_form,
    ),
    "brightness-temperature": Quantity(
        "brightness-temperature",
        "Brightness temperature",
        ssa_tools.TB_energy_form,
        ssa_tools.TB_frequency_form,
    ),
    "gamma-min-constraint": Quantity(
        "gamma-min-constraint",
        "Gamma-min constraint",
        ssa_tools.gamma_min_constraint,
        None,
    ),
}


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults while preserving manually formatted help sections."""


def quantity_choices_text() -> str:
    return ", ".join(["all"] + sorted(QUANTITIES.keys()))


def full_help_epilog() -> str:
    return textwrap.dedent(
        f"""
        Run command options:
          input_yaml
              YAML file containing parameter distributions.
          --form {{auto,energy,frequency}}
              Calculation form. Use auto to infer from YAML keys.
          --quantity, -q QUANTITY [QUANTITY ...]
              Quantities to calculate. Choices: {quantity_choices_text()}.
          --samples, -n SAMPLES
              Number of Monte Carlo samples per quantity.
          --seed SEED
              Random seed for reproducible sampling.
          --ci-level CI_LEVEL
              Central confidence interval level.
          --linear
              Return linear values instead of log10 values.
          --output-format {{table,json}}
              File results format. Terminal output is always a table.
          --output, -o OUTPUT
              Write an extra results file. Defaults to JSON when a file is written.
          --run-name RUN_NAME
              Save a named run under --runs-dir with input.yaml, results, metadata,
              and plots unless --no-plots is passed.
          --runs-dir RUNS_DIR
              Parent directory for named runs.
          --overwrite
              Replace an existing named run directory.
          --plot-dir PLOT_DIR
              Write posterior PDFs and one priors.pdf to this directory. With
              --run-name, defaults to RUNS_DIR/RUN_NAME/plots.
          --no-plots
              Do not save plots for a named run.
          --bins BINS
              Number of histogram bins for diagnostic plots.

        Other commands:
          list-quantities
              Show available quantity names and supported forms.

        Examples:
          ssa-estimates run inputs.yaml
          ssa-estimates run inputs.yaml --run-name first-pass
          ssa-estimates run inputs.yaml --quantity energy radius --samples 10000 --run-name source-a
        """
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssa-estimates",
        description=(
            "Run synchrotron self-absorption estimates from a YAML parameter "
            "file."
        ),
        epilog=full_help_epilog(),
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="run Monte Carlo estimates from a YAML parameter file",
        formatter_class=HelpFormatter,
    )
    run_parser.add_argument(
        "input_yaml",
        type=Path,
        help="path to a YAML file containing parameter distributions",
    )
    run_parser.add_argument(
        "--form",
        choices=("auto", "energy", "frequency"),
        default="auto",
        help="calculation form to use; default infers this from the YAML keys",
    )
    run_parser.add_argument(
        "--quantity",
        "-q",
        nargs="+",
        default=["all"],
        choices=["all"] + sorted(QUANTITIES.keys()),
        metavar="QUANTITY",
        help=f"one or more quantities to calculate. Choices: {quantity_choices_text()}",
    )
    run_parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=1000,
        help="number of Monte Carlo samples per quantity",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible sampling",
    )
    run_parser.add_argument(
        "--ci-level",
        type=float,
        default=0.68,
        help="central confidence interval level",
    )
    run_parser.add_argument(
        "--linear",
        action="store_true",
        help="return linear values instead of the default log10 values",
    )
    run_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="json",
        help="file output format; terminal output is always a table",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write an extra results file to this path",
    )
    run_parser.add_argument(
        "--run-name",
        help=(
            "save a named run under --runs-dir with input.yaml, results, "
            "metadata, and plots"
        ),
    )
    run_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="parent directory for named run folders",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing named run directory",
    )
    run_parser.add_argument(
        "--plot-dir",
        type=Path,
        help=(
            "write posterior diagnostic PDFs to this directory; with "
            "--run-name, defaults to RUNS_DIR/RUN_NAME/plots"
        ),
    )
    run_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="do not save plots for a named run",
    )
    run_parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="number of histogram bins for diagnostic plots",
    )
    run_parser.set_defaults(func=run_command)

    list_parser = subparsers.add_parser(
        "list-quantities",
        help="show available quantity names",
        formatter_class=HelpFormatter,
    )
    list_parser.set_defaults(func=list_quantities_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    command_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command_argv)
    args.command_argv = command_argv
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


def list_quantities_command(args: argparse.Namespace) -> int:
    del args
    for key in sorted(QUANTITIES):
        quantity = QUANTITIES[key]
        forms = []
        if quantity.energy_function is not None:
            forms.append("energy")
        if quantity.frequency_function is not None:
            forms.append("frequency")
        print(f"{key:24s} {', '.join(forms)}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.no_plots and args.plot_dir is not None:
        raise SystemExit("--no-plots cannot be used with --plot-dir")

    params, config = load_parameter_file(args.input_yaml)
    form = resolve_form(args.form, params, config)
    quantity_names = resolve_quantity_names(args.quantity, form)
    run_bundle = prepare_run_bundle(args)
    plot_dir = resolve_plot_dir(args, run_bundle)

    if run_bundle is not None:
        copy_input_file(args.input_yaml, run_bundle.input_path)

    results = []
    priors_saved = False
    for quantity_name in quantity_names:
        quantity = QUANTITIES[quantity_name]
        function = get_function_for_form(quantity, form)
        function_params = params_for_function(function, params)
        result = mc_uncertainty(
            function,
            function_params,
            n=args.samples,
            seed=args.seed,
            ci_level=args.ci_level,
            fixed_kwargs={"log10": not args.linear},
        )
        record = result_to_record(
            quantity=quantity,
            form=form,
            result=result,
            ci_level=args.ci_level,
            log10=not args.linear,
            samples=args.samples,
        )
        results.append(record)

        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"{quantity.key}.pdf"
            priors_path = None
            if not priors_saved:
                priors_path = plot_dir / "priors.pdf"
                priors_saved = True

            plot_mc_diagnostics(
                result,
                function_params,
                ci_level=args.ci_level,
                bins=args.bins,
                save_path=str(plot_path),
                save_priors_path=str(priors_path) if priors_path is not None else None,
                show=False,
            )

    print(format_results(results, "table"))

    result_paths = result_output_paths(args, run_bundle)
    if result_paths:
        file_text = format_results(results, args.output_format)
        for output_path in result_paths:
            write_text_file(output_path, file_text)

    if run_bundle is not None:
        write_run_metadata(
            run_bundle=run_bundle,
            args=args,
            form=form,
            quantity_names=quantity_names,
            params=params,
            result_paths=result_paths,
            plot_dir=plot_dir,
        )
    return 0


def prepare_run_bundle(args: argparse.Namespace) -> Optional[RunBundle]:
    if args.run_name is None:
        return None

    run_name = sanitize_run_name(args.run_name)
    run_dir = args.runs_dir / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Run directory already exists: {run_dir}. "
                "Pass --overwrite to replace it."
            )
        if not run_dir.is_dir():
            raise SystemExit(f"Run path exists and is not a directory: {run_dir}")
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    results_name = "results.json" if args.output_format == "json" else "results.txt"
    plots_dir = None if args.no_plots else (args.plot_dir or run_dir / "plots")
    return RunBundle(
        name=run_name,
        directory=run_dir,
        input_path=run_dir / "input.yaml",
        metadata_path=run_dir / "run_metadata.json",
        results_path=run_dir / results_name,
        plots_dir=plots_dir,
    )


def sanitize_run_name(run_name: str) -> str:
    cleaned = run_name.strip().replace("/", "-").replace("\\", "-")
    cleaned = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "-"
        for char in cleaned
    ).strip("-._")
    if not cleaned:
        raise SystemExit("--run-name must contain at least one letter or number")
    return cleaned


def resolve_plot_dir(
    args: argparse.Namespace,
    run_bundle: Optional[RunBundle],
) -> Optional[Path]:
    if args.no_plots:
        return None
    if args.plot_dir is not None:
        return args.plot_dir
    if run_bundle is not None:
        return run_bundle.plots_dir
    return None


def copy_input_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def result_output_paths(
    args: argparse.Namespace,
    run_bundle: Optional[RunBundle],
) -> List[Path]:
    paths: List[Path] = []
    if run_bundle is not None:
        paths.append(run_bundle.results_path)
    if args.output is not None:
        paths.append(args.output)

    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)
    return unique_paths


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_run_metadata(
    run_bundle: RunBundle,
    args: argparse.Namespace,
    form: str,
    quantity_names: Sequence[str],
    params: Mapping[str, Any],
    result_paths: Sequence[Path],
    plot_dir: Optional[Path],
) -> None:
    metadata = {
        "run_name": run_bundle.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ssa_estimates_version": __version__,
        "command": ["ssa-estimates"] + list(getattr(args, "command_argv", [])),
        "input_file": str(args.input_yaml),
        "saved_input_file": str(run_bundle.input_path),
        "run_directory": str(run_bundle.directory),
        "form": form,
        "quantities": list(quantity_names),
        "parameter_names": sorted(params.keys()),
        "samples": args.samples,
        "seed": args.seed,
        "ci_level": args.ci_level,
        "scale": "linear" if args.linear else "log10",
        "output_format": args.output_format,
        "result_files": [str(path) for path in result_paths],
        "plot_dir": str(plot_dir) if plot_dir is not None else None,
        "plots_enabled": plot_dir is not None,
    }
    write_text_file(run_bundle.metadata_path, json.dumps(metadata, indent=2, sort_keys=True))


def load_parameter_file(path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, Mapping):
        raise SystemExit(f"{path} must contain a YAML mapping")

    if "parameters" in data:
        params = data["parameters"]
        config = {k: v for k, v in data.items() if k != "parameters"}
    else:
        params = data
        config = {}

    if not isinstance(params, Mapping):
        raise SystemExit("'parameters' must be a YAML mapping")

    return dict(params), dict(config)


def resolve_form(
    requested_form: str,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    if requested_form != "auto":
        return requested_form

    configured_form = config.get("form")
    if configured_form in ("energy", "frequency"):
        return str(configured_form)
    if configured_form is not None:
        raise SystemExit("YAML 'form' must be 'energy' or 'frequency'")

    has_energy_keys = {"gamma_min", "gamma_max"}.issubset(params)
    has_frequency_keys = {"nu_min", "nu_max"}.issubset(params)

    if has_energy_keys and not has_frequency_keys:
        return "energy"
    if has_frequency_keys and not has_energy_keys:
        return "frequency"
    if has_energy_keys and has_frequency_keys:
        raise SystemExit("YAML contains both energy and frequency keys; pass --form")
    raise SystemExit(
        "Could not infer calculation form. Add gamma_min/gamma_max, "
        "nu_min/nu_max, a top-level form key, or pass --form."
    )


def resolve_quantity_names(requested: Iterable[str], form: str) -> List[str]:
    requested = list(requested)
    if "all" in requested:
        return [
            name
            for name in sorted(QUANTITIES)
            if get_function_for_form(QUANTITIES[name], form) is not None
        ]
    else:
        names = requested

    unsupported = [
        name
        for name in names
        if get_function_for_form(QUANTITIES[name], form) is None
    ]
    if unsupported:
        joined = ", ".join(unsupported)
        raise SystemExit(f"Quantity not available for {form} form: {joined}")
    return names


def get_function_for_form(
    quantity: Quantity,
    form: str,
) -> Optional[Callable[..., float]]:
    if form == "energy":
        return quantity.energy_function
    if form == "frequency":
        return quantity.frequency_function
    raise ValueError(f"Unexpected form: {form}")


def params_for_function(
    function: Callable[..., float],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    signature = inspect.signature(function)
    required_names = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in required_names if name not in params]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required YAML parameter(s): {joined}")
    return {name: params[name] for name in required_names}


def result_to_record(
    quantity: Quantity,
    form: str,
    result: Any,
    ci_level: float,
    log10: bool,
    samples: int,
) -> Dict[str, Any]:
    return {
        "quantity": quantity.key,
        "label": quantity.label,
        "form": form,
        "scale": "log10" if log10 else "linear",
        "samples_requested": samples,
        "samples_used": int(len(result.samples)),
        "mean": result.mean,
        "median": result.median,
        "std": result.std,
        "ci_level": ci_level,
        "ci_low": result.ci[0],
        "ci_high": result.ci[1],
    }


def format_results(results: Sequence[Mapping[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(list(results), indent=2, sort_keys=True)
    if output_format != "table":
        raise ValueError(f"Unexpected output format: {output_format}")

    headers = [
        "quantity",
        "form",
        "scale",
        "mean",
        "median",
        "std",
        "ci_low",
        "ci_high",
        "n_used",
    ]
    rows = [
        [
            str(row["quantity"]),
            str(row["form"]),
            str(row["scale"]),
            format_number(float(row["mean"])),
            format_number(float(row["median"])),
            format_number(float(row["std"])),
            format_number(float(row["ci_low"])),
            format_number(float(row["ci_high"])),
            str(row["samples_used"]),
        ]
        for row in results
    ]
    return format_table(headers, rows)


def format_number(value: float) -> str:
    return f"{value:.6g}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(value)) for value in column)
        for column in zip(headers, *rows)
    ]
    header = "  ".join(value.ljust(width) for value, width in zip(headers, widths))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in rows
    ]
    return "\n".join([header, separator] + body)


if __name__ == "__main__":
    sys.exit(main())
