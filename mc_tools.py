#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:54:19 2026

@author: cowie
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional, Sequence, List, Union, Any
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import warnings

# --- Types & results ---------------------------------------------------------
ParamSpec = Union[Tuple[float, float], Dict[str, Any]]

@dataclass
class MCResult:
    mean: float
    std: float
    median: float
    ci: Tuple[float, float]
    samples: np.ndarray                # posterior samples of y
    param_samples: Dict[str, np.ndarray]  # raw draws per parameter (same length as samples)

# --- Spec handling & sampling -----------------------------------------------
def _to_lognormal_log_params(mean: float, sigma: float) -> Tuple[float, float]:
    if mean <= 0:
        raise ValueError("Lognormal 'mean' must be > 0 when space='linear'.")
    if sigma < 0:
        raise ValueError("Lognormal 'sigma' must be >= 0.")
    if sigma == 0.0:
        return float(np.log(mean)), 0.0
    sigma2 = sigma**2
    mu_log = np.log(mean**2 / np.sqrt(sigma2 + mean**2))
    sigma_log = np.sqrt(np.log(1.0 + sigma2 / (mean**2)))
    return float(mu_log), float(sigma_log)

# --- Spec handling & sampling -----------------------------------------------
def _resolve_spec(spec: ParamSpec) -> Dict[str, Any]:
    if isinstance(spec, tuple):
        m, s = spec
        return {
            "dist": "normal",
            "sampler_params": (float(m), float(s)),
            "space": "normal",
            "bounds": None,
            "nominal": float(m),
            "strict_positive": False,
            "eps": 1e-300,
        }

    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError("Spec must be tuple or dict with a 'dist' key.")

    dist = spec["dist"].lower()
    bounds = spec.get("bounds", None)
    strict_positive = bool(spec.get("strict_positive", False))
    eps = float(spec.get("eps", 1e-300))

    if dist == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if not (high > low):
            raise ValueError("For uniform, require high > low.")
        space = spec.get("space", "linear").lower()
        if space == "linear":
            nominal = 0.5 * (low + high)
        elif space == "log":
            if low <= 0:
                raise ValueError("Log-uniform requires low > 0.")
            nominal = np.sqrt(low * high)  # geometric mean
        else:
            raise ValueError("space must be 'linear' or 'log' for uniform.")
        return {
            "dist": "uniform",
            "sampler_params": (low, high),
            "space": space,
            "bounds": bounds,
            "nominal": nominal,
            "strict_positive": strict_positive,
            "eps": eps,
        }

    if dist == "normal":
        m = float(spec["mean"])
        s = float(spec["sigma"]) if "sigma" in spec else float(abs(m) * spec["sigma_frac"])
        return {"dist": "normal", "sampler_params": (m, s), "space": "normal", "bounds": bounds,
                "nominal": m, "strict_positive": strict_positive, "eps": eps}

    if dist == "lognormal":
        space = spec.get("space", "linear").lower()
        if space == "linear":
            m = float(spec["mean"])
            s = float(spec["sigma"]) if "sigma" in spec else float(abs(m) * spec["sigma_frac"])
            mu_log, sigma_log = _to_lognormal_log_params(m, s)
            nominal = m
        elif space == "log":
            mu_log = float(spec["mean"])
            sigma_log = float(spec["sigma"])
            nominal = float(np.exp(mu_log))
        else:
            raise ValueError("space must be 'linear' or 'log' for lognormal.")
        return {"dist": "lognormal", "sampler_params": (mu_log, sigma_log), "space": "log", "bounds": bounds,
                "nominal": nominal, "strict_positive": True, "eps": eps}

    # --- New: asymmetric (split) normal ---
    if dist in ("asymm_normal", "asymmetric_normal", "split_normal", "two_piece_normal"):
        m = float(spec["mean"])
        # Prefer explicit sigma_up/down if provided, otherwise use sigma_frac if present.
        if "sigma_up" in spec or "sigma_down" in spec:
            if "sigma_up" not in spec or "sigma_down" not in spec:
                raise ValueError("asymm_normal requires both 'sigma_up' and 'sigma_down' if one is provided.")
            sigma_up = float(spec["sigma_up"])
            sigma_down = float(spec["sigma_down"])
        else:
            if "sigma_frac" not in spec:
                raise ValueError("asymm_normal requires 'sigma_up' and 'sigma_down', or 'sigma_frac' to set both.")
            sigma_up = float(abs(m) * spec["sigma_frac"])
            sigma_down = float(abs(m) * spec["sigma_frac"])
        if sigma_up < 0 or sigma_down < 0:
            raise ValueError("Asymmetric normal sigmas must be >= 0.")
        # Store as (mean, sigma_down, sigma_up) to be consistent when sampling
        return {"dist": "asymm_normal", "sampler_params": (m, sigma_down, sigma_up), "space": "normal",
                "bounds": bounds, "nominal": m, "strict_positive": strict_positive, "eps": eps}

    raise ValueError("Unsupported 'dist'.")

def _sample_param(rng: np.random.Generator, resolved: Dict[str, Any], n: int) -> np.ndarray:
    dist = resolved["dist"]
    lo_hi = resolved.get("bounds", None)
    eps = float(resolved.get("eps", 1e-300))

    if dist == "normal":
        mu, sigma = resolved["sampler_params"]
        if sigma < 0:
            raise ValueError("Sigma must be >= 0.")
        x = rng.normal(mu, sigma, size=n)

    elif dist == "lognormal":
        mu_log, sigma_log = resolved["sampler_params"]
        if sigma_log < 0:
            raise ValueError("Sigma must be >= 0.")
        x = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)

    elif dist == "uniform":
        low, high = resolved["sampler_params"]
        if resolved["space"] == "linear":
            x = rng.uniform(low, high, size=n)
        elif resolved["space"] == "log":
            loglow, loghigh = np.log10(low), np.log10(high)
            logx = rng.uniform(loglow, loghigh, size=n)
            x = np.power(10.0, logx)
        else:
            raise RuntimeError("Unexpected space for uniform.")

    elif dist == "asymm_normal":
        # Split/Two-piece normal: draw standard normal and scale positive/negative sides differently
        mu, sigma_down, sigma_up = resolved["sampler_params"]
        if (sigma_down < 0) or (sigma_up < 0):
            raise ValueError("Asymmetric normal sigmas must be >= 0.")
        z = rng.normal(0.0, 1.0, size=n)
        # For z >= 0 use sigma_up, for z < 0 use sigma_down
        scales = np.where(z >= 0.0, sigma_up, sigma_down)
        x = mu + z * scales

    else:
        raise RuntimeError("Unexpected dist.")

    if lo_hi is not None:
        lo, hi = lo_hi
        if lo is not None:
            x = np.maximum(x, lo)
        if hi is not None:
            x = np.minimum(x, hi)

    if resolved.get("strict_positive", False):
        x = np.where(x <= eps, eps, x)

    return x

# --- Monte Carlo core --------------------------------------------------------
def mc_uncertainty(
    f: Callable[..., float],
    params: Dict[str, ParamSpec],
    n: int = 100_000,
    seed: Optional[int] = None,
    ci_level: float = 0.68,
    fixed_kwargs: Optional[Dict[str, Any]] = None,
) -> MCResult:
    """
    Monte Carlo error propagation for y = f(**params), with per-parameter
    normal / lognormal / uniform / asymmetric-normal uncertainties.

    This version evaluates f one sample at a time, which is safer for
    functions that call scalar solvers such as scipy.optimize.root.
    """
    rng = np.random.default_rng(seed)

    if fixed_kwargs is None:
        fixed_kwargs = {}

    # Resolve distributions and draw samples
    resolved = {k: _resolve_spec(v) for k, v in params.items()}
    draws = {k: _sample_param(rng, resolved[k], n) for k in resolved.keys()}

    # Evaluate sample-by-sample
    y = np.empty(n, dtype=np.complex128)
    for i in range(n):
        kwargs_i = {k: v[i] for k, v in draws.items()}
        y[i] = f(**kwargs_i, **fixed_kwargs)

    # Warn if any complex values were produced
    imag = np.abs(np.imag(y))
    n_complex = int(np.sum(imag > 0))
    if n_complex > 0:
        warnings.warn(
            f"mc_uncertainty: {n_complex} samples returned complex values; "
            f"their imaginary parts will be discarded."
        )

    # Keep only the real part for downstream statistics
    y = np.real(y)

    # Remove invalid values
    mask = np.isfinite(y)
    n_valid = int(np.sum(mask))
    if n_valid < n:
        warnings.warn(
            f"mc_uncertainty: {n - n_valid} invalid samples were discarded; "
            f"effective sample size = {n_valid}."
        )

    y = y[mask]
    draws = {k: v[mask] for k, v in draws.items()}

    if len(y) == 0:
        raise RuntimeError("All Monte Carlo samples were invalid.")

    mean = float(np.mean(y))
    median = float(np.median(y))
    std = float(np.std(y, ddof=1))
    lo, hi = np.quantile(y, [(1 - ci_level) / 2, 1 - (1 - ci_level) / 2])

    return MCResult(
        mean=mean,
        std=std,
        ci=(float(lo), float(hi)),
        samples=y,
        param_samples=draws,
        median=median,
    )

# --- Plotting utilities ------------------------------------------------------
def plot_mc_diagnostics(
    result,
    params,
    ci_level: float = 0.68,
    bins: int = 60,
    max_cols: int = 3,
):
    """
    Make quick-look plots:
      • A grid of histograms (one per input parameter).
        - Normal parameters: linear histogram
        - Lognormal parameters: histogram in log10 space
      • A separate histogram for the posterior result.
    """
    # Resolve again to know which parameters are lognormal
    from copy import deepcopy
    from types import SimpleNamespace
    
    # reuse your _resolve_spec
    resolved = {k: _resolve_spec(v) for k, v in params.items()}
    names = list(result.param_samples.keys())
    K = len(names)

    # --- Parameter samples grid ---
    rows = math.ceil(K / max_cols)
    fig_params, axes = plt.subplots(rows, min(K, max_cols),
                                    figsize=(5*min(K, max_cols), 3.6*rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, -1)

    for idx, name in enumerate(names):
        r, c = divmod(idx, max_cols)
        ax = axes[r, c]
        x = result.param_samples[name]
        spec = resolved[name]

        # treat asymmetric normal same as normal for plotting
        if spec["dist"] == "lognormal" or (spec["dist"] == "uniform" and spec["space"] == "log"):
            # Plot in log10 space
            logx = np.log10(x)
            ax.hist(logx, bins=bins, color="C0", alpha=0.7)
            ax.set_xlabel("log10(" + name + ")")
            nominal = np.log10(spec["nominal"])
            ax.axvline(nominal, color="k")
        else:
            ax.hist(x, bins=bins, color="C0", alpha=0.7)
            ax.set_xlabel(name)
            ax.axvline(spec["nominal"], color="k")

        ax.set_title(name)
        ax.set_ylabel("count")

    # Hide unused panels
    # for j in range(K, rows*max_cols):
    #     r, c = divmod(j, max_cols)
    #     axes[r, c].axis("off")

    fig_params.tight_layout()
    
    y = result.samples
    lo2, hi2 = np.quantile(y, [(1 - 0.95) / 2, 1 - (1 - 0.95) / 2])
    lo3, hi3 = np.quantile(y, [(1 - 0.997) / 2, 1 - (1 - 0.997) / 2])

    # --- Posterior histogram ---
    y = result.samples
    lo, hi = result.ci
    fig_post, axp = plt.subplots(figsize=(10, 6))
    axp.hist(y, bins=bins, color="C1", alpha=0.7, density=False)
    axp.axvline(result.mean, color="k", label="mean")
    axp.axvline(result.median, color="c", label="median")
    axp.axvline(lo, linestyle="--", color="k", label=f"{int(ci_level*100)}% CI")
    axp.axvline(hi, linestyle="--", color="k")
    #axp.set_title(f"Posterior of result (mean={result.mean:.4g}, std={result.std:.4g})")
    # axp.set_xlabel("log(Minimum energy [erg])")
    axp.set_xlabel("Quantity")
    axp.set_ylabel("Counts")
    #axp.set_xlim(40,50)
    axp.legend()
    axp.text(0.02, 0.95, f"{int(ci_level*100)}% CI: [{lo:.4g}, {hi:.4g}]",
             transform=axp.transAxes, va="top")
    axp.text(0.02, 0.88, f"{int(0.95*100)}% CI: [{lo2:.4g}, {hi2:.4g}]",
             transform=axp.transAxes, va="top")
    axp.text(0.02, 0.81, f"{(0.997*100):.3g}% CI: [{lo3:3.4g}, {hi3:.4g}]",
             transform=axp.transAxes, va="top")


    # mu, sigma, D, p = fit_normal_and_gof(res.samples)
    # xfit = np.linspace(min(res.samples), max(res.samples), 200)
    # axp.plot(xfit, norm.pdf(xfit, mu, sigma), 'r--', label='Log-normal fit')
    
    fig_post.tight_layout()
    
    plt.savefig('/Users/cowie/Downloads/mc_13.pdf', dpi=300, bbox_inches='tight')
    plt.show() #Had to add this line in for some reason

    return {"params": fig_params, "posterior": fig_post}