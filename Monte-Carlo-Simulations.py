"""
credit_mc_simulation.py
Monte Carlo Simulation Engine for Credit Derivatives Strategy Analysis
Author: [YOUR NAME]
Date: February 2026

This module implements:
  1. CIR-based stochastic spread simulation for IG and HY indices
  2. Gaussian copula default simulation for a 125-name CDO portfolio
  3. CDS and payer swaption P&L computation
  4. CLO equity cash flow waterfall and IRR estimation
  5. Visualisation routines

All parameters are exposed as configurable inputs. Replace placeholder
values with live market data before running.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL CONFIGURATION -- REPLACE PLACEHOLDERS WITH LIVE MARKET DATA
# ============================================================================

CONFIG = {
    # --- Current market levels (as of February 19, 2026, from FRED) ---
    "ig_spread_bps": 79.0,         # ICE BofA IG OAS (bps)
    "hy_spread_bps": 288.0,        # ICE BofA HY OAS (bps)
    "cdx_ig_spread_bps": 54.0,     # Estimated: cash minus ~25 bps basis
    "cdx_hy_spread_bps": 283.0,    # Estimated: cash minus ~5 bps
    "risk_free_rate": 0.0415,      # 10Y UST yield
    "fed_funds_rate": 0.03625,     # Mid of 3.50-3.75% range
    "recovery_rate": 0.40,         # Standard senior unsecured recovery
    "ig_oad": 6.79,                # IG option-adjusted duration (years)
    "hy_oad": 3.30,                # HY option-adjusted duration (years)
    "vix": 17.0,                   # VIX level
    "move_index": 66.0,            # MOVE index level

    # --- CIR process parameters (fitted from FRED data, 2021-02-19 to 2026-02-19) ---
    "ig_kappa": 0.3922,            # Mean reversion speed (IG)
    "ig_theta_bps": 100.4,         # Long-run mean spread (IG, bps)
    "ig_sigma": 2.4322,            # Vol of vol (IG)
    "hy_kappa": 1.6566,            # Mean reversion speed (HY)
    "hy_theta_bps": 353.9,         # Long-run mean spread (HY, bps)
    "hy_sigma": 6.9479,            # Vol of vol (HY)
    "ig_hy_correlation": 0.7492,   # From daily spread change correlation

    # --- Copula / portfolio parameters ---
    "n_obligors": 125,             # CDX IG reference portfolio size
    "copula_rho": 0.25,            # Gaussian copula correlation (academic consensus)
    "avg_default_prob_1y": 0.013167, # Implied: 79 bps / (1 - 0.40) = 131.7 bps hazard

    # --- CLO parameters ---
    "clo_aaa_spread_bps": 125.0,   # AAA tranche spread over SOFR
    "clo_loan_spread_bps": 350.0,  # Average leveraged loan spread
    "clo_equity_pct": 0.10,        # Equity tranche as % of deal
    "clo_reinvest_years": 4,       # Reinvestment period
    "clo_maturity_years": 8,       # Legal maturity

    # --- Simulation settings ---
    "n_simulations": 50000,
    "n_time_steps": 252,           # Daily steps for 1 year
    "horizon_years": 1.0,
    "random_seed": 42,
}

# Scenario definitions: (drift_multiplier, vol_multiplier, label)
SCENARIOS = {
    "base": {
        "ig_theta_mult": 1.0,
        "hy_theta_mult": 1.0,
        "vol_mult": 1.0,
        "default_mult": 1.0,
        "label": "Base Case (Soft Landing)",
        "color": "#2ecc71",
        "weight": 0.50,
    },
    "moderate_widening": {
        "ig_theta_mult": 1.5,      # Mean reverts toward 195 bps
        "hy_theta_mult": 1.3,      # Mean reverts toward 637 bps
        "vol_mult": 1.5,
        "default_mult": 2.0,
        "label": "Moderate Widening (Mild Recession)",
        "color": "#f39c12",
        "weight": 0.35,
    },
    "severe_stress": {
        "ig_theta_mult": 3.0,      # Mean reverts toward 390 bps
        "hy_theta_mult": 2.5,      # Mean reverts toward 1225 bps
        "vol_mult": 3.0,
        "default_mult": 5.0,
        "label": "Severe Stress (GFC-like)",
        "color": "#e74c3c",
        "weight": 0.15,
    },
}


# ============================================================================
# 1. CIR SPREAD SIMULATION
# ============================================================================

def simulate_cir_spread(s0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    """
    Simulate credit spread paths using the CIR process:
        ds = kappa * (theta - s) * dt + sigma * sqrt(s) * dW

    Parameters
    ----------
    s0 : float       Initial spread (bps)
    kappa : float     Mean reversion speed
    theta : float     Long-run mean (bps)
    sigma : float     Volatility parameter
    T : float         Time horizon (years)
    n_steps : int     Number of time steps
    n_paths : int     Number of simulation paths
    rng : np.random.Generator

    Returns
    -------
    paths : ndarray of shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0

    for i in range(n_steps):
        s = paths[:, i]
        s_pos = np.maximum(s, 0.0)  # Ensure non-negativity
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        ds = kappa * (theta - s_pos) * dt + sigma * np.sqrt(s_pos) * dW
        paths[:, i + 1] = np.maximum(s_pos + ds, 0.01)  # Floor at 0.01 bps

    return paths


def simulate_correlated_spreads(cfg, scenario, rng):
    """
    Simulate correlated IG and HY spread paths under a given scenario.

    Returns
    -------
    ig_paths, hy_paths : ndarray of shape (n_sims, n_steps + 1)
    """
    sc = SCENARIOS[scenario]
    n = cfg["n_simulations"]
    steps = cfg["n_time_steps"]
    T = cfg["horizon_years"]
    rho = cfg["ig_hy_correlation"]

    # Generate correlated Brownian increments
    Z1 = rng.standard_normal((n, steps))
    Z2 = rng.standard_normal((n, steps))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    dt = T / steps

    # IG paths
    ig = np.zeros((n, steps + 1))
    ig[:, 0] = cfg["ig_spread_bps"]
    kappa_ig = cfg["ig_kappa"]
    theta_ig = cfg["ig_theta_bps"] * sc["ig_theta_mult"]
    sigma_ig = cfg["ig_sigma"] * sc["vol_mult"]

    for i in range(steps):
        s = np.maximum(ig[:, i], 0.01)
        ig[:, i+1] = np.maximum(
            s + kappa_ig * (theta_ig - s) * dt + sigma_ig * np.sqrt(s * dt) * W1[:, i],
            0.01
        )

    # HY paths
    hy = np.zeros((n, steps + 1))
    hy[:, 0] = cfg["hy_spread_bps"]
    kappa_hy = cfg["hy_kappa"]
    theta_hy = cfg["hy_theta_bps"] * sc["hy_theta_mult"]
    sigma_hy = cfg["hy_sigma"] * sc["vol_mult"]

    for i in range(steps):
        s = np.maximum(hy[:, i], 0.01)
        hy[:, i+1] = np.maximum(
            s + kappa_hy * (theta_hy - s) * dt + sigma_hy * np.sqrt(s * dt) * W2[:, i],
            0.01
        )

    return ig, hy


# ============================================================================
# 2. GAUSSIAN COPULA DEFAULT SIMULATION
# ============================================================================

def simulate_defaults_copula(n_obligors, n_sims, rho, default_prob, rng):
    """
    Simulate correlated defaults using the one-factor Gaussian copula.

    X_i = sqrt(rho) * M + sqrt(1 - rho) * eps_i
    Default if X_i < Phi^{-1}(p_i)

    Parameters
    ----------
    n_obligors : int
    n_sims : int
    rho : float        Asset correlation
    default_prob : float  Marginal default probability (1Y)
    rng : np.random.Generator

    Returns
    -------
    n_defaults : ndarray of shape (n_sims,)  Number of defaults per sim
    loss_pct : ndarray of shape (n_sims,)    Portfolio loss percentage
    """
    threshold = norm.ppf(default_prob)

    # Systematic factor
    M = rng.standard_normal(n_sims)

    # Idiosyncratic factors
    eps = rng.standard_normal((n_sims, n_obligors))

    # Latent variables
    X = np.sqrt(rho) * M[:, np.newaxis] + np.sqrt(1 - rho) * eps

    # Defaults
    defaults = (X < threshold).astype(int)
    n_defaults = defaults.sum(axis=1)

    # Portfolio loss (equal weight, fixed LGD)
    lgd = 1.0 - CONFIG["recovery_rate"]
    loss_pct = (n_defaults / n_obligors) * lgd * 100  # As percentage

    return n_defaults, loss_pct


def tranche_loss(portfolio_loss_pct, attach, detach):
    """
    Compute tranche loss given portfolio loss percentage.

    Parameters
    ----------
    portfolio_loss_pct : ndarray  Portfolio loss as % of notional
    attach : float                Attachment point (e.g., 0.03 for 3%)
    detach : float                Detachment point (e.g., 0.07 for 7%)

    Returns
    -------
    tranche_loss_pct : ndarray    Tranche loss as % of tranche notional
    """
    port_loss = portfolio_loss_pct / 100.0  # Convert to fraction
    width = detach - attach
    tranche_loss = np.clip(port_loss - attach, 0, width) / width * 100
    return tranche_loss


# ============================================================================
# 3. CDS AND PAYER SWAPTION P&L
# ============================================================================

def cds_pnl_from_spread_change(notional, spread_init_bps, spread_final_bps,
                                risky_dv01_per_mm=4.5):
    """
    Approximate mark-to-market P&L on a CDS position from spread change.
    Protection buyer profits when spreads widen.

    Parameters
    ----------
    notional : float          Notional in USD
    spread_init_bps : float   Entry spread
    spread_final_bps : float  Exit spread
    risky_dv01_per_mm : float Risky DV01 per $1M notional (years)

    Returns
    -------
    pnl : float  P&L in USD (positive = profit for protection buyer)
    """
    delta_spread = spread_final_bps - spread_init_bps
    pnl = (notional / 1e6) * risky_dv01_per_mm * delta_spread * 100
    return pnl


def payer_swaption_payoff(forward_spread, strike, rpv01, impl_vol, T,
                           notional, is_payer=True):
    """
    Price a CDS index swaption using Black's model.

    Payer = RPV01 * [F*N(d1) - K*N(d2)]
    Receiver = RPV01 * [K*N(-d2) - F*N(-d1)]

    Parameters
    ----------
    forward_spread : float  Forward CDS spread (bps)
    strike : float          Strike spread (bps)
    rpv01 : float           Risky PV01 (per bp per $1M notional)
    impl_vol : float        Implied spread volatility (e.g., 0.42 for 42%)
    T : float               Time to expiry (years)
    notional : float        Notional in USD
    is_payer : bool         True for payer (protection buyer)

    Returns
    -------
    premium : float         Option premium in USD
    """
    if forward_spread <= 0 or strike <= 0:
        return 0.0

    d1 = (np.log(forward_spread / strike) + 0.5 * impl_vol**2 * T) / (impl_vol * np.sqrt(T))
    d2 = d1 - impl_vol * np.sqrt(T)

    if is_payer:
        price_per_bp = forward_spread * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        price_per_bp = strike * norm.cdf(-d2) - forward_spread * norm.cdf(-d1)

    premium = rpv01 * price_per_bp * (notional / 1e6)
    return premium


def simulate_payer_swaption_pnl(ig_terminal_spreads, cfg):
    """
    Compute P&L distribution for a payer swaption on CDX IG.

    Parameters
    ----------
    ig_terminal_spreads : ndarray  Terminal IG spread for each simulation
    cfg : dict                     Configuration

    Returns
    -------
    pnl : ndarray  P&L per simulation (USD, for $10M notional)
    """
    notional = 10_000_000
    strike = cfg["cdx_ig_spread_bps"] + 30  # 30 bps OTM payer
    rpv01 = 4.5  # Per $1M notional
    impl_vol = 0.42
    T = 0.25  # 3-month expiry

    # Premium paid upfront
    premium = payer_swaption_payoff(
        cfg["cdx_ig_spread_bps"], strike, rpv01, impl_vol, T, notional, True
    )

    # Payoff at expiry: max(S_T - K, 0) * RPV01
    intrinsic = np.maximum(ig_terminal_spreads - strike, 0)
    payoff = rpv01 * intrinsic * (notional / 1e6)

    pnl = payoff - premium
    return pnl


# ============================================================================
# 4. CLO EQUITY CASH FLOW MODEL
# ============================================================================

def clo_equity_irr(loan_spread_path_bps, cfg, rng):
    """
    Simplified CLO equity cash flow model.

    Computes annual equity distributions from the excess spread between
    loan portfolio income and rated tranche costs, net of defaults.

    Parameters
    ----------
    loan_spread_path_bps : ndarray  Annual avg loan spread for each year
    cfg : dict
    rng : np.random.Generator

    Returns
    -------
    irr : float  Internal rate of return for equity tranche
    """
    maturity = cfg["clo_maturity_years"]
    reinvest = cfg["clo_reinvest_years"]
    equity_pct = cfg["clo_equity_pct"]
    aaa_cost = cfg["clo_aaa_spread_bps"] / 10000
    base_rate = cfg["fed_funds_rate"]

    # Assume deal size $500M, equity = $50M
    deal_size = 500_000_000
    equity_invest = deal_size * equity_pct

    # Simplified capital structure costs (blended rated tranche cost)
    # AAA: 65% at S+125, AA: 10% at S+150, A: 8% at S+190,
    # BBB: 5% at S+300, BB: 2% at S+600
    blended_liability_spread = (
        0.65 * 125 + 0.10 * 150 + 0.08 * 190 + 0.05 * 300 + 0.02 * 600
    ) / 10000  # Convert to decimal

    # Annual default rate assumption
    annual_default_rate = 0.02 + rng.uniform(-0.01, 0.02)
    recovery = cfg["recovery_rate"]

    cash_flows = [-equity_invest]

    for yr in range(1, maturity + 1):
        if yr <= reinvest:
            loan_spread = loan_spread_path_bps[min(yr-1, len(loan_spread_path_bps)-1)] / 10000
        else:
            # Amortisation period: use last reinvestment spread
            loan_spread = loan_spread_path_bps[min(reinvest-1, len(loan_spread_path_bps)-1)] / 10000

        # Gross income
        performing_pct = (1 - annual_default_rate) ** yr
        gross_income = deal_size * performing_pct * (base_rate + loan_spread)

        # Liability costs
        liability_cost = deal_size * (1 - equity_pct) * (base_rate + blended_liability_spread)

        # Default losses absorbed by equity first
        default_loss = deal_size * annual_default_rate * (1 - recovery) * performing_pct

        # Excess spread to equity
        excess = gross_income - liability_cost - default_loss

        # Management fees (~50bps of AUM)
        mgmt_fee = deal_size * performing_pct * 0.005

        equity_cf = max(excess - mgmt_fee, 0)

        # Terminal: return of equity principal (simplified)
        if yr == maturity:
            principal_return = max(equity_invest * performing_pct - default_loss * 2, 0)
            equity_cf += principal_return

        cash_flows.append(equity_cf)

    # Compute IRR
    try:
        irr = brentq(lambda r: sum(cf / (1 + r)**i for i, cf in enumerate(cash_flows)),
                      -0.5, 2.0)
    except (ValueError, RuntimeError):
        irr = np.nan

    return irr


# ============================================================================
# 5. MASTER SIMULATION RUNNER
# ============================================================================

def run_full_simulation(cfg=CONFIG):
    """Run the complete Monte Carlo simulation across all scenarios."""
    rng = np.random.default_rng(cfg["random_seed"])
    results = {}

    for scenario_name, sc in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Running scenario: {sc['label']}")
        print(f"{'='*60}")

        # --- 5a. Simulate correlated IG/HY spread paths ---
        ig_paths, hy_paths = simulate_correlated_spreads(cfg, scenario_name, rng)

        ig_terminal = ig_paths[:, -1]
        hy_terminal = hy_paths[:, -1]

        print(f"  IG terminal spread: mean={ig_terminal.mean():.1f}, "
              f"5th={np.percentile(ig_terminal, 5):.1f}, "
              f"95th={np.percentile(ig_terminal, 95):.1f} bps")
        print(f"  HY terminal spread: mean={hy_terminal.mean():.1f}, "
              f"5th={np.percentile(hy_terminal, 5):.1f}, "
              f"95th={np.percentile(hy_terminal, 95):.1f} bps")

        # --- 5b. CDS P&L (protection buyer on $10M notional) ---
        cds_pnl = np.array([
            cds_pnl_from_spread_change(10_000_000, cfg["cdx_ig_spread_bps"], s)
            for s in ig_terminal
        ])
        print(f"  CDS protection buyer P&L: mean=${cds_pnl.mean():,.0f}, "
              f"median=${np.median(cds_pnl):,.0f}")

        # --- 5c. Payer swaption P&L ---
        swaption_pnl = simulate_payer_swaption_pnl(ig_terminal, cfg)
        print(f"  Payer swaption P&L: mean=${swaption_pnl.mean():,.0f}, "
              f"95th pctile=${np.percentile(swaption_pnl, 95):,.0f}")

        # --- 5d. Default simulation (copula) ---
        adj_default_prob = cfg["avg_default_prob_1y"] * sc["default_mult"]
        n_def, loss_pct = simulate_defaults_copula(
            cfg["n_obligors"], cfg["n_simulations"],
            cfg["copula_rho"], adj_default_prob, rng
        )
        print(f"  Portfolio defaults: mean={n_def.mean():.2f}, "
              f"max={n_def.max()}")
        print(f"  Portfolio loss: mean={loss_pct.mean():.3f}%, "
              f"99th={np.percentile(loss_pct, 99):.3f}%")

        # Tranche losses
        equity_loss = tranche_loss(loss_pct, 0.00, 0.03)
        mezz_loss = tranche_loss(loss_pct, 0.03, 0.07)
        senior_loss = tranche_loss(loss_pct, 0.07, 0.15)

        print(f"  Equity tranche (0-3%) loss: mean={equity_loss.mean():.2f}%")
        print(f"  Mezz tranche (3-7%) loss: mean={mezz_loss.mean():.2f}%")
        print(f"  Senior tranche (7-15%) loss: mean={senior_loss.mean():.2f}%")

        # --- 5e. CLO equity IRR (subsample for speed) ---
        n_clo_sims = min(1000, cfg["n_simulations"])
        clo_irrs = []
        for j in range(n_clo_sims):
            # Use simulated HY spread path as proxy for loan spreads
            annual_spreads = hy_paths[j, ::cfg["n_time_steps"] // cfg["clo_maturity_years"]]
            if len(annual_spreads) < cfg["clo_maturity_years"]:
                annual_spreads = np.pad(annual_spreads,
                    (0, cfg["clo_maturity_years"] - len(annual_spreads)),
                    mode="edge")
            irr = clo_equity_irr(annual_spreads[:cfg["clo_maturity_years"]], cfg, rng)
            clo_irrs.append(irr)
        clo_irrs = np.array(clo_irrs)
        valid_irrs = clo_irrs[~np.isnan(clo_irrs)]
        if len(valid_irrs) > 0:
            print(f"  CLO equity IRR: mean={valid_irrs.mean()*100:.1f}%, "
                  f"median={np.median(valid_irrs)*100:.1f}%")

        results[scenario_name] = {
            "ig_paths": ig_paths,
            "hy_paths": hy_paths,
            "ig_terminal": ig_terminal,
            "hy_terminal": hy_terminal,
            "cds_pnl": cds_pnl,
            "swaption_pnl": swaption_pnl,
            "n_defaults": n_def,
            "loss_pct": loss_pct,
            "equity_tranche_loss": equity_loss,
            "mezz_tranche_loss": mezz_loss,
            "senior_tranche_loss": senior_loss,
            "clo_irrs": clo_irrs,
        }

    return results


# ============================================================================
# 6. VISUALISATION
# ============================================================================

def plot_spread_paths(results, cfg, n_sample=200):
    """Figure 1: Simulated IG and HY spread paths under three scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
    t = np.linspace(0, cfg["horizon_years"], cfg["n_time_steps"] + 1)

    for j, (sname, sc) in enumerate(SCENARIOS.items()):
        r = results[sname]
        # IG
        for i in range(min(n_sample, r["ig_paths"].shape[0])):
            axes[0, j].plot(t, r["ig_paths"][i], alpha=0.03, color=sc["color"], lw=0.5)
        axes[0, j].plot(t, np.median(r["ig_paths"], axis=0), color="black", lw=2,
                        label="Median")
        axes[0, j].plot(t, np.percentile(r["ig_paths"], 95, axis=0), color="black",
                        lw=1, ls="--", label="95th pctile")
        axes[0, j].axhline(cfg["ig_spread_bps"], color="grey", ls=":", lw=1)
        axes[0, j].set_title(f"IG Spread: {sc['label']}", fontsize=11)
        axes[0, j].set_xlabel("Years")
        axes[0, j].set_ylabel("Spread (bps)")
        axes[0, j].legend(fontsize=8)

        # HY
        for i in range(min(n_sample, r["hy_paths"].shape[0])):
            axes[1, j].plot(t, r["hy_paths"][i], alpha=0.03, color=sc["color"], lw=0.5)
        axes[1, j].plot(t, np.median(r["hy_paths"], axis=0), color="black", lw=2)
        axes[1, j].plot(t, np.percentile(r["hy_paths"], 95, axis=0), color="black",
                        lw=1, ls="--")
        axes[1, j].axhline(cfg["hy_spread_bps"], color="grey", ls=":", lw=1)
        axes[1, j].set_title(f"HY Spread: {sc['label']}", fontsize=11)
        axes[1, j].set_xlabel("Years")
        axes[1, j].set_ylabel("Spread (bps)")

    plt.suptitle("Figure 1: Simulated Credit Spread Paths Under Three Macro Scenarios",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figure1_spread_paths.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_swaption_pnl(results, cfg):
    """Figure 2: P&L distribution of payer swaption strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for j, (sname, sc) in enumerate(SCENARIOS.items()):
        pnl = results[sname]["swaption_pnl"]
        axes[j].hist(pnl, bins=80, color=sc["color"], alpha=0.7, edgecolor="white",
                     density=True)
        axes[j].axvline(0, color="black", ls="--", lw=1)
        axes[j].axvline(pnl.mean(), color="red", ls="-", lw=2,
                        label=f"Mean: ${pnl.mean():,.0f}")
        pct95 = np.percentile(pnl, 95)
        axes[j].axvline(pct95, color="darkred", ls="--", lw=1.5,
                        label=f"95th: ${pct95:,.0f}")
        axes[j].set_title(sc["label"], fontsize=11)
        axes[j].set_xlabel("P&L ($)")
        axes[j].legend(fontsize=9)
        # Mark % profitable
        pct_profit = (pnl > 0).mean() * 100
        axes[j].text(0.95, 0.95, f"{pct_profit:.0f}% profitable",
                     transform=axes[j].transAxes, ha="right", va="top", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("Density")
    plt.suptitle("Figure 2: Payer Swaption P&L Distribution ($10M Notional, 3M Expiry, 30bps OTM)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figure2_swaption_pnl.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_portfolio_loss(results, cfg):
    """Figure 3: Portfolio loss distribution from Gaussian copula."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    tranches = [
        ("equity_tranche_loss", "Equity (0-3%)", "#e74c3c"),
        ("mezz_tranche_loss", "Mezzanine (3-7%)", "#f39c12"),
        ("senior_tranche_loss", "Senior (7-15%)", "#2ecc71"),
    ]

    for j, (sname, sc) in enumerate(SCENARIOS.items()):
        r = results[sname]
        axes[j].hist(r["loss_pct"], bins=60, color=sc["color"], alpha=0.7,
                     edgecolor="white", density=True, label="Portfolio")
        axes[j].axvline(r["loss_pct"].mean(), color="black", lw=2,
                        label=f"Mean: {r['loss_pct'].mean():.3f}%")
        axes[j].axvline(np.percentile(r["loss_pct"], 99), color="red", lw=1.5, ls="--",
                        label=f"99th: {np.percentile(r['loss_pct'], 99):.3f}%")
        axes[j].set_title(sc["label"], fontsize=11)
        axes[j].set_xlabel("Portfolio Loss (%)")
        axes[j].legend(fontsize=8)

    axes[0].set_ylabel("Density")
    plt.suptitle("Figure 3: Portfolio Loss Distribution (125-Name CDX IG, Gaussian Copula)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figure3_portfolio_loss.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_clo_equity_irr(results, cfg):
    """Figure 4: CLO equity IRR distribution across scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for j, (sname, sc) in enumerate(SCENARIOS.items()):
        irrs = results[sname]["clo_irrs"]
        valid = irrs[~np.isnan(irrs)] * 100  # Convert to %
        if len(valid) == 0:
            axes[j].text(0.5, 0.5, "No valid IRRs", transform=axes[j].transAxes,
                         ha="center")
            continue
        axes[j].hist(valid, bins=50, color=sc["color"], alpha=0.7, edgecolor="white",
                     density=True)
        axes[j].axvline(valid.mean(), color="black", lw=2,
                        label=f"Mean: {valid.mean():.1f}%")
        axes[j].axvline(np.median(valid), color="blue", lw=1.5, ls="--",
                        label=f"Median: {np.median(valid):.1f}%")
        axes[j].axvline(np.percentile(valid, 10), color="red", lw=1.5, ls=":",
                        label=f"10th: {np.percentile(valid, 10):.1f}%")
        axes[j].set_title(sc["label"], fontsize=11)
        axes[j].set_xlabel("IRR (%)")
        axes[j].legend(fontsize=9)

    axes[0].set_ylabel("Density")
    plt.suptitle("Figure 4: CLO Equity IRR Distribution ($500M Deal, 10% Equity)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figure4_clo_equity_irr.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_strategy_comparison(results, cfg):
    """Figure 5: Comparative P&L across strategies (probability-weighted)."""
    strategies = ["Long CDS Protection", "Payer Swaption (30bp OTM)", "IG/HY Decompression"]
    weighted_pnl = {s: np.zeros(cfg["n_simulations"]) for s in strategies}

    for sname, sc in SCENARIOS.items():
        r = results[sname]
        w = sc["weight"]

        # Long CDS protection
        weighted_pnl[strategies[0]] += w * r["cds_pnl"]

        # Payer swaption
        weighted_pnl[strategies[1]] += w * r["swaption_pnl"]

        # IG/HY decompression: long IG prot + short HY prot
        decomp_pnl = r["cds_pnl"] - np.array([
            cds_pnl_from_spread_change(10_000_000, cfg["cdx_hy_spread_bps"], s, 3.3)
            for s in r["hy_terminal"]
        ])
        weighted_pnl[strategies[2]] += w * decomp_pnl

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    positions = []
    for i, (strat, pnl) in enumerate(weighted_pnl.items()):
        bp = ax.boxplot(pnl, positions=[i], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=colors[i], alpha=0.6),
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(lw=1.5),
                        flierprops=dict(marker=".", markersize=1, alpha=0.3))
        ax.text(i, np.percentile(pnl, 75) + 5000,
                f"Mean: ${pnl.mean():,.0f}\nSharpe: {pnl.mean()/pnl.std():.2f}",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=11)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_ylabel("P&L ($)", fontsize=12)
    ax.set_title("Figure 5: Probability-Weighted Strategy P&L Comparison ($10M Notional)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figure5_strategy_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CREDIT DERIVATIVES MONTE CARLO SIMULATION ENGINE")
    print(f"Simulations: {CONFIG['n_simulations']:,}")
    print(f"Horizon: {CONFIG['horizon_years']} year(s)")
    print(f"Scenarios: {', '.join(sc['label'] for sc in SCENARIOS.values())}")
    print("=" * 60)

    # Run simulation
    results = run_full_simulation(CONFIG)

    # Generate all figures
    print("\nGenerating visualisations...")
    plot_spread_paths(results, CONFIG)
    plot_swaption_pnl(results, CONFIG)
    plot_portfolio_loss(results, CONFIG)
    plot_clo_equity_irr(results, CONFIG)
    plot_strategy_comparison(results, CONFIG)

    # Summary statistics
    print("\n" + "=" * 60)
    print("PROBABILITY-WEIGHTED SUMMARY")
    print("=" * 60)
    for strat_name, key in [("CDS Protection Buyer", "cds_pnl"),
                             ("Payer Swaption", "swaption_pnl")]:
        weighted = sum(
            SCENARIOS[s]["weight"] * results[s][key]
            for s in SCENARIOS
        )
        print(f"\n{strat_name} ($10M notional):")
        print(f"  Expected P&L:   ${weighted.mean():>12,.0f}")
        print(f"  Std Dev:        ${weighted.std():>12,.0f}")
        print(f"  Sharpe Ratio:   {weighted.mean()/weighted.std():>12.3f}")
        print(f"  5th percentile: ${np.percentile(weighted, 5):>12,.0f}")
        print(f"  95th percentile:${np.percentile(weighted, 95):>12,.0f}")
        print(f"  Max Loss:       ${weighted.min():>12,.0f}")
        print(f"  Max Gain:       ${weighted.max():>12,.0f}")
        pct_profit = (weighted > 0).mean() * 100
        print(f"  % Profitable:   {pct_profit:>11.1f}%")

    print("\nSimulation complete. Figures saved to current directory.")
    print("Replace CONFIG placeholders with live data for production use.")