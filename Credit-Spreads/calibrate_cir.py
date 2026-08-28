"""
calibrate_cir.py
Reads FRED CSV files for IG and HY OAS, fits CIR parameters,
computes correlation, and prints CONFIG values to paste into
credit_mc_simulation.py.

USAGE:
  1. Download CSVs from FRED:
     - https://fred.stlouisfed.org/series/BAMLC0A0CM  (IG OAS)
     - https://fred.stlouisfed.org/series/BAMLH0A0HYM2 (HY OAS)
  2. Save them in the same folder as this script.
     FRED names them: BAMLC0A0CM.csv and BAMLH0A0HYM2.csv
  3. Run: python calibrate_cir.py
  4. Copy the printed CONFIG values into credit_mc_simulation.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ============================================================================
# 1. LOAD DATA
# ============================================================================

def load_fred_csv(filepath, col_name):
    """Load a FRED CSV. FRED format: DATE, VALUE with '.' for missing."""
    df = pd.read_csv(filepath)
    # FRED CSVs have columns like 'DATE' and the series name
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df = df.rename(columns={"value": col_name})
    return df

print("Loading FRED data...")
print("  Looking for BAMLC0A0CM.csv (IG OAS) and BAMLH0A0HYM2.csv (HY OAS)")
print()

try:
    ig_df = load_fred_csv("BAMLC0A0CM.csv", "ig_oas")
except FileNotFoundError:
    print("ERROR: BAMLC0A0CM.csv not found.")
    print("Download it from: https://fred.stlouisfed.org/series/BAMLC0A0CM")
    print("Click 'Download' -> CSV, save in this folder.")
    exit(1)

try:
    hy_df = load_fred_csv("BAMLH0A0HYM2.csv", "hy_oas")
except FileNotFoundError:
    print("ERROR: BAMLH0A0HYM2.csv not found.")
    print("Download it from: https://fred.stlouisfed.org/series/BAMLH0A0HYM2")
    print("Click 'Download' -> CSV, save in this folder.")
    exit(1)

# Merge on date
df = pd.merge(ig_df, hy_df, on="date", how="inner")

# FRED OAS is in percentage points (e.g., 0.78 = 78 bps). Convert to bps.
# Check: if max IG value < 20, it's in percentage points, multiply by 100
if df["ig_oas"].max() < 20:
    df["ig_oas"] = df["ig_oas"] * 100  # Convert to bps
if df["hy_oas"].max() < 50:
    df["hy_oas"] = df["hy_oas"] * 100

print(f"Loaded {len(df)} daily observations")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Latest IG OAS: {df['ig_oas'].iloc[-1]:.1f} bps")
print(f"Latest HY OAS: {df['hy_oas'].iloc[-1]:.1f} bps")
print()

# ============================================================================
# 2. FIT CIR PARAMETERS BY MAXIMUM LIKELIHOOD
# ============================================================================

def fit_cir_mle(spread_series, dt=1/252):
    """
    Fit CIR process parameters (kappa, theta, sigma) to a time series
    of spread levels using conditional MLE.

    The CIR process: ds = kappa*(theta - s)*dt + sigma*sqrt(s)*dW

    For the discrete approximation, the conditional distribution of
    s_{t+1} given s_t is approximately normal with:
      mean = s_t + kappa*(theta - s_t)*dt
      var  = sigma^2 * s_t * dt

    We maximise the log-likelihood of observed transitions.
    """
    s = spread_series.values
    s = s[s > 0]  # Remove any zeros
    n = len(s) - 1

    s_t = s[:-1]   # s at time t
    s_t1 = s[1:]   # s at time t+1
    ds = s_t1 - s_t

    def neg_log_likelihood(params):
        kappa, theta, sigma = params
        if kappa <= 0 or theta <= 0 or sigma <= 0:
            return 1e12
        # Feller condition check (soft penalty)
        if 2 * kappa * theta < sigma**2:
            penalty = 1000 * (sigma**2 - 2 * kappa * theta)
        else:
            penalty = 0

        mu = kappa * (theta - s_t) * dt
        var = sigma**2 * s_t * dt
        var = np.maximum(var, 1e-10)

        # Gaussian log-likelihood for each transition
        ll = -0.5 * np.log(2 * np.pi * var) - 0.5 * (ds - mu)**2 / var
        return -np.sum(ll) + penalty

    # Initial guesses
    theta0 = np.mean(s)
    kappa0 = 0.5
    sigma0 = np.std(ds) / np.sqrt(np.mean(s) * dt)

    result = minimize(
        neg_log_likelihood,
        x0=[kappa0, theta0, sigma0],
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8}
    )

    kappa, theta, sigma = result.x
    return abs(kappa), abs(theta), abs(sigma)


print("Fitting CIR parameters (this takes a few seconds)...")
print()

ig_kappa, ig_theta, ig_sigma = fit_cir_mle(df["ig_oas"])
hy_kappa, hy_theta, hy_sigma = fit_cir_mle(df["hy_oas"])

print("IG CIR Parameters:")
print(f"  kappa (mean reversion speed): {ig_kappa:.4f}")
print(f"  theta (long-run mean, bps):   {ig_theta:.1f}")
print(f"  sigma (vol of vol):           {ig_sigma:.4f}")
feller_ig = 2 * ig_kappa * ig_theta / ig_sigma**2
print(f"  Feller ratio (>1 = positive): {feller_ig:.2f}")
print()

print("HY CIR Parameters:")
print(f"  kappa (mean reversion speed): {hy_kappa:.4f}")
print(f"  theta (long-run mean, bps):   {hy_theta:.1f}")
print(f"  sigma (vol of vol):           {hy_sigma:.4f}")
feller_hy = 2 * hy_kappa * hy_theta / hy_sigma**2
print(f"  Feller ratio (>1 = positive): {feller_hy:.2f}")
print()

# ============================================================================
# 3. COMPUTE IG-HY CORRELATION
# ============================================================================

ig_changes = df["ig_oas"].diff().dropna()
hy_changes = df["hy_oas"].diff().dropna()
correlation = ig_changes.corr(hy_changes)

print(f"IG-HY daily spread change correlation: {correlation:.4f}")
print()

# ============================================================================
# 4. DERIVE DEFAULT PROBABILITY
# ============================================================================

latest_ig = df["ig_oas"].iloc[-1]
latest_hy = df["hy_oas"].iloc[-1]
recovery = 0.40
ig_hazard = (latest_ig / 10000) / (1 - recovery)  # Convert bps to decimal
hy_hazard = (latest_hy / 10000) / (1 - recovery)

print(f"Implied IG hazard rate: {ig_hazard*10000:.1f} bps ({ig_hazard*100:.3f}% p.a.)")
print(f"Implied HY hazard rate: {hy_hazard*10000:.1f} bps ({hy_hazard*100:.3f}% p.a.)")
print()

# ============================================================================
# 5. PRINT CONFIG VALUES TO COPY-PASTE
# ============================================================================

print("=" * 60)
print("COPY THESE INTO credit_mc_simulation.py CONFIG:")
print("=" * 60)
print()
print("# --- Current market levels ---")
print(f'"ig_spread_bps": {latest_ig:.1f},')
print(f'"hy_spread_bps": {latest_hy:.1f},')
print(f'"cdx_ig_spread_bps": {max(latest_ig - 25, 30):.1f},  # Estimated: cash minus ~25 bps basis')
print(f'"cdx_hy_spread_bps": {latest_hy - 5:.1f},  # Estimated: cash minus ~5 bps')
print()
print("# --- CIR parameters (fitted from FRED data) ---")
print(f'"ig_kappa": {ig_kappa:.4f},')
print(f'"ig_theta_bps": {ig_theta:.1f},')
print(f'"ig_sigma": {ig_sigma:.4f},')
print(f'"hy_kappa": {hy_kappa:.4f},')
print(f'"hy_theta_bps": {hy_theta:.1f},')
print(f'"hy_sigma": {hy_sigma:.4f},')
print(f'"ig_hy_correlation": {correlation:.4f},')
print()
print("# --- Default probability (derived from spread) ---")
print(f'"avg_default_prob_1y": {ig_hazard:.6f},')
print()
print("=" * 60)
print("Done. Paste the above into your CONFIG dictionary.")