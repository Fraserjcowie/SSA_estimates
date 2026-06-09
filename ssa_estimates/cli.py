"""Command-line interface for SSA estimates."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from . import ssa_tools
from .mc_tools import mc_uncertainty, plot_mc_diagnostics


@dataclass(frozen=True)
class Quantity:
    key: str
    label: str
    energy_function: Optional[Callable[..., float]]
    frequency_function: Optional[Callable[..., float]]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssa-estimates",
        description=(
            "Run synchrotron self-absorption estimates from a YAML parameter "
            "file."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="run Monte Carlo estimates from a YAML parameter file",
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
        help="one or more quantities to calculate",
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
        default="table",
        help="output format",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="write results to this file instead of stdout",
    )
    run_parser.add_argument(
        "--plot-dir",
        type=Path,
        help="write posterior diagnostic PDFs to this directory",
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
    )
    list_parser.set_defaults(func=list_quantities_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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

    params, config = load_parameter_file(args.input_yaml)
    form = resolve_form(args.form, params, config)
    quantity_names = resolve_quantity_names(args.quantity, form)

    results = []
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

        if args.plot_dir is not None:
            args.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = args.plot_dir / f"{quantity.key}.pdf"
            plot_mc_diagnostics(
                result,
                function_params,
                ci_level=args.ci_level,
                bins=args.bins,
                save_path=str(plot_path),
                show=False,
            )

    text = format_results(results, args.output_format)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


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
