# SSA Estimates

If you make use of this software, please cite:

```bibtex
@ARTICLE{2026MNRAS.tmp.1046C,
       author = {{Cowie}, F.~J. and {Fender}, R.~P.},
        title = "{Towards improved synchrotron self absorption energy estimates: accounting for inhomogeneous and non-spherical emitting regions}",
      journal = {\mnras},
     keywords = {High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = jun,
          doi = {10.1093/mnras/stag1113},
archivePrefix = {arXiv},
       eprint = {2606.11307},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026MNRAS.tmp.1046C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
---

The command line tool, `ssa-estimates`, provided by this repo calculates physical quantities from observations of the synchrotron self-absorption peak. The key observables required are the frequency and amplitude of the spectral peak. 
The full required inputs are given through a YAML file, where uncertainties can be included by specifying input parameter distributions to draw values from.

The mathematical details of the framework can be found in the Cowie & Fender 2026 (details above). In order to extract physical quantities such as the internal energy, magnetic field, 
and emitting region size, the assumptions of homogeneity, quasi-sphericity, and a power law non-thermal electron energy spectrum are made. For more details on the effects of relaxing these assumptions see Cowie & Fender 2026.

The package currently supports two formalisms which require slightly different information to calcualte the desired physical parameters:

- `energy`: the energy formalism uses `gamma_min` (the minimum cut-off in the electron energy spectrum) and `gamma_max` (the maximum cut-off in the electron energy spectrum).
- `frequency`: the frequency formalism uses `nu_min` (the minimum charecteristic frequency of synchrotron radiating electrons) and `nu_max` (the maximum charecteristic frequency of synchrotron radiating electrons).

`ssa-estimates` runs Monte Carlo uncertainty propagation and reports 
summary statistics for the requested quantities.

## Installation

Clone the repository and then from the repository root run:

```bash
python -m pip install .
```

This installs the `ssa-estimates` command.

For editable development installs, use `python -m pip install -e .` with a
modern `pip` and `setuptools`.

## Quick Start

Calculate all quantities from the example input file in the energy formalism:

```bash
ssa-estimates run inputs.yaml
```

Calcualte one quantity with reproducible sampling and JSON output:

```bash
ssa-estimates run inputs.yaml \
  --quantity energy \
  --samples 10000 \
  --seed 100 \
  --output results.json
```

The `run` command always prints a table to the terminal with the results. When results are written to
a file with `--output` or `--run-name`, the default file format is JSON.

Save a complete named run:

```bash
ssa-estimates run inputs.yaml \
  --run-name source-a-first-pass \
  --quantity energy radius \
  --samples 10000 \
  --seed 100
```

This creates a directory structure of:

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

Calculate all quantities from the example input file in the frequency formalism:

```bash
ssa-estimates run inputs_frequency.yaml
```

List available quantities which can be calculated:

```bash
ssa-estimates list-quantities
```

## Input YAML

The example input files `inputs.yaml` and `inputs_frequency.yaml` use the simplest
supported layout: a mapping from parameter names to probability distributions.
Parameter names must match the calculation function arguments.
The example files can be easily edited to contain the information relevant for your case.

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

You can also use a structured layout with an explicit formalism:

```yaml
form: energy
parameters:
  flux_dens_peak_mJy:
    dist: normal
    mean: 839.4
    sigma: 0.3
```

Supported prior distributions from which samples are drawn for the Monte Carlo are:

- `normal`
- `lognormal`
- `uniform`
- `asymm_normal`

Options for the distributions such as `bounds` (include bounds on the prior distributions), `strict_positive` (ensure that samples are positive), and `sigma_frac` (specify sigma as a fraction rather than an absolute value).
See the example YAML files for complete parameter examples.

## Calculated Quantities

Use these names with `--quantity`:

- `energy` - the internal energy of the emitting region
- `radius` - the intrinsic (radial) size of the emitting region
- `magnetic-field` - the magnetic field of the emitting region
- `n0` - the normalisation for the electron energy distribution
- `electron-density` - the density of the non-thermal electrons in the emitting region
- `electron-number` - the total number of electrons in the emitting reigon
- `brightness-temperature` - the rest frame (intrinsic) brightness temperature of the emitting region
- `gamma-min-constraint` - the upper limit for the minimum electron energy cut-off (only applicable in the energy formalism)

By default, results are reported in `log10` space. Pass `--linear` to report
linear values for all calcualted quantities.

## Diagnostics

To save posterior diagnostic plots:

```bash
ssa-estimates run inputs.yaml --quantity energy --plot-dir plots
```

Each calculated quantity has a posterior in the plot directory. The command
also writes one `priors.pdf` file containing all prior parameter distributions
sampled for the MC run.

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
- `plots/`: one `priors.pdf` plus one posterior per requested quantity.

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
