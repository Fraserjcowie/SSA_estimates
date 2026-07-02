# SSA Estimates

`ssa-estimates` calculates physicsal quantities from observations of the synchrotron self-absorption peak. The key observables are the frequency and amplitude of the spectral peak. The full required inputs are given through a YAML
file, where uncertainties can be included by specifying input parameter distributions to draw values from.

The package currently supports two calculation forms:

- `energy`: uses `gamma_min` and `gamma_max`.
- `frequency`: uses `nu_min` and `nu_max`.

The command line tool runs Monte Carlo uncertainty propagation and reports
summary statistics for the requested quantities.

## Install Locally

From the repository root:

```bash
python -m pip install .
```

This installs the `ssa-estimates` command.

For editable development installs, use `python -m pip install -e .` with a
modern `pip` and `setuptools`.

## Quick Start

Run all available energy-form quantities from the example input file:

```bash
ssa-estimates run inputs.yaml
```

Run one quantity with reproducible sampling and JSON output:

```bash
ssa-estimates run inputs.yaml \
  --quantity energy \
  --samples 10000 \
  --seed 100 \
  --output results.json
```

The command always prints a table to the terminal. When results are written to
a file with `--output` or `--run-name`, the default file format is JSON.

Save a complete named run:

```bash
ssa-estimates run inputs.yaml \
  --run-name source-a-first-pass \
  --quantity energy radius \
  --samples 10000 \
  --seed 100
```

This creates:

```text
runs/
  source-a-first-pass/
    input.yaml
    results.json
    run_metadata.json
    plots/
      priors.pdf
      energy.pdf
      radius.pdf
```

Run the frequency-form example:

```bash
ssa-estimates run inputs_frequency.yaml
```

List available quantities:

```bash
ssa-estimates list-quantities
```

## Input YAML

The example files `inputs.yaml` and `inputs_frequency.yaml` use the simplest
supported layout: a mapping from parameter names to probability distributions.
Parameter names must match the calculation function arguments.

```yaml
flux_dens_peak_mJy:
  dist: normal
  mean: 839.4
  sigma: 0.3

dist_kpc:
  dist: uniform
  low: 1.5
  high: 4.3
```

You can also use a structured layout with an explicit form:

```yaml
form: energy
parameters:
  flux_dens_peak_mJy:
    dist: normal
    mean: 839.4
    sigma: 0.3
```

Supported distributions are:

- `normal`
- `lognormal`
- `uniform`
- `asymm_normal`

Common optional fields include `bounds`, `strict_positive`, and `sigma_frac`.
See the example YAML files for complete parameter examples.

## Quantities

Use these names with `--quantity`:

- `energy`
- `radius`
- `magnetic-field`
- `n0`
- `electron-density`
- `electron-number`
- `brightness-temperature`
- `gamma-min-constraint` for the energy form only

By default, results are reported in `log10` space. Pass `--linear` to report
linear values.

## Diagnostics

To save posterior diagnostic plots:

```bash
ssa-estimates run inputs.yaml --quantity energy --plot-dir plots
```

Each requested quantity gets a posterior PDF in the plot directory. The command
also writes one `priors.pdf` file containing all prior parameter distributions
sampled for the run.

## Named Runs

Use `--run-name` to save a run as one folder under `runs/`:

```bash
ssa-estimates run inputs.yaml --run-name source-a
```

The run folder contains:

- `input.yaml`: an exact copy of the input file used for the run.
- `results.json`: machine-readable result summaries by default.
- `run_metadata.json`: form, quantities, sample count, seed, confidence level,
  output scale, package version, command, and output paths.
- `plots/`: one `priors.pdf` plus one posterior PDF per requested quantity.

Useful options:

- `--runs-dir PATH`: choose a different parent directory for named runs.
- `--overwrite`: replace an existing run folder with the same name.
- `--no-plots`: save inputs/results/metadata without generating PDFs.
- `--output-format table`: write saved result files as a table instead of JSON.
- `--output PATH`: write an additional result file outside the run folder.

## Development

Run tests with:

```bash
python -m unittest discover -s tests
```
