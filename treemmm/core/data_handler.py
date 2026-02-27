"""Data ingestion, validation, panel balancing, and diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from treemmm.core.config import ColumnSpec, Objective, RunConfig, TemporalAlignment


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class DistributionDiagnostic:
    """Result of the outcome distribution analysis."""

    n_obs: int
    n_zeros: int
    zero_frac: float
    is_integer: bool
    skewness: float
    mean: float
    variance: float
    var_mean_ratio: float
    recommended_objective: Objective
    reasoning: str


@dataclass
class ReverseCausalityResult:
    """Result of the reverse-causality diagnostic for one promo variable."""

    variable: str
    granger_p_value: float
    lead_test_p_value: float
    flagged: bool
    recommendation: TemporalAlignment


@dataclass
class PanelDiagnostic:
    """Summary of panel balance and quality checks."""

    n_customers: int
    n_periods: int
    expected_rows: int
    actual_rows: int
    missing_rows: int
    customers_complete: int
    customers_incomplete: int
    zero_variance_cols: list[str]
    high_vif_cols: list[tuple[str, float]]


@dataclass
class PreparedData:
    """Validated, balanced, feature-engineered panel ready for modeling."""

    df: pd.DataFrame
    config: RunConfig
    panel_diagnostic: PanelDiagnostic
    distribution_diagnostic: DistributionDiagnostic
    reverse_causality_results: list[ReverseCausalityResult]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load data from a DataFrame, CSV path, or Parquet path."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Distribution diagnostic
# ---------------------------------------------------------------------------
def diagnose_distribution(series: pd.Series) -> DistributionDiagnostic:
    """Analyse outcome column to recommend an objective function.

    Checks discreteness, zero-inflation, skewness, and mean-variance
    relationship to recommend gaussian / poisson / tweedie / gamma.
    """
    vals = series.dropna().values.astype(float)
    n_obs = len(vals)
    n_zeros = int(np.sum(vals == 0))
    zero_frac = n_zeros / n_obs if n_obs > 0 else 0.0
    is_integer = bool(np.all(vals == np.floor(vals)))
    skewness = float(sp_stats.skew(vals)) if n_obs > 2 else 0.0
    mean_val = float(np.mean(vals))
    var_val = float(np.var(vals, ddof=1)) if n_obs > 1 else 0.0
    vmr = var_val / mean_val if mean_val > 0 else float("inf")

    # Decision logic
    if is_integer and (vals >= 0).all():
        if zero_frac > 0.3:
            rec = Objective.TWEEDIE
            reason = (
                f"Integer non-negative outcome with {zero_frac:.0%} zeros. "
                "Tweedie handles zero-inflation and count-like structure."
            )
        elif vmr > 2.0:
            rec = Objective.POISSON
            reason = (
                f"Integer non-negative outcome with var/mean={vmr:.1f} (overdispersed). "
                "Poisson objective with log-link is appropriate for count data."
            )
        else:
            rec = Objective.POISSON
            reason = (
                "Integer non-negative outcome with moderate dispersion. "
                "Poisson objective recommended for count data."
            )
    elif (vals >= 0).all() and zero_frac > 0.2:
        rec = Objective.TWEEDIE
        reason = (
            f"Non-negative continuous outcome with {zero_frac:.0%} zeros. "
            "Tweedie handles point mass at zero + positive continuous tail."
        )
    elif (vals > 0).all():
        if skewness > 1.5:
            rec = Objective.GAMMA
            reason = (
                f"Strictly positive, right-skewed (skew={skewness:.2f}). "
                "Gamma with log-link models multiplicative effects."
            )
        else:
            rec = Objective.GAUSSIAN
            reason = (
                "Continuous, roughly symmetric or mildly skewed. "
                "Gaussian / MSE is appropriate."
            )
    else:
        rec = Objective.GAUSSIAN
        reason = (
            "Outcome contains negative values or mixed structure. "
            "Defaulting to Gaussian / MSE."
        )

    return DistributionDiagnostic(
        n_obs=n_obs,
        n_zeros=n_zeros,
        zero_frac=zero_frac,
        is_integer=is_integer,
        skewness=skewness,
        mean=mean_val,
        variance=var_val,
        var_mean_ratio=vmr,
        recommended_objective=rec,
        reasoning=reason,
    )


# ---------------------------------------------------------------------------
# Reverse causality diagnostic
# ---------------------------------------------------------------------------
def _granger_test(
    outcome: np.ndarray,
    promo: np.ndarray,
    max_lag: int,
) -> float:
    """Simple panel Granger causality test (pooled OLS F-test).

    Tests whether lagged promo values improve prediction of outcome
    beyond lagged outcome values alone.  Returns minimum p-value across
    lag orders 1..max_lag.
    """
    from sklearn.linear_model import LinearRegression

    best_p = 1.0
    n = len(outcome)
    for lag in range(1, min(max_lag + 1, n // 3)):
        y = outcome[lag:]
        # Restricted model: lagged outcome only
        x_r = np.column_stack([outcome[lag - k - 1 : n - k - 1] for k in range(lag)])
        # Unrestricted model: lagged outcome + lagged promo
        x_u = np.column_stack(
            [x_r] + [promo[lag - k - 1 : n - k - 1] for k in range(lag)]
        )

        if x_u.shape[0] < x_u.shape[1] + 2:
            continue

        lr_r = LinearRegression().fit(x_r, y)
        lr_u = LinearRegression().fit(x_u, y)
        ssr_r = float(np.sum((y - lr_r.predict(x_r)) ** 2))
        ssr_u = float(np.sum((y - lr_u.predict(x_u)) ** 2))

        df_num = lag  # extra parameters in unrestricted
        df_den = len(y) - x_u.shape[1]
        if df_den <= 0 or ssr_u <= 0:
            continue

        f_stat = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_den)
        p_val = 1.0 - sp_stats.f.cdf(f_stat, df_num, df_den)
        best_p = min(best_p, p_val)

    return best_p


def _lead_test(outcome: np.ndarray, promo: np.ndarray) -> float:
    """Test if future promo predicts current outcome (reverse causality signal).

    If promo_{t+1} correlates with outcome_t after controlling for
    outcome_{t-1}, this suggests promo is allocated *in response to*
    outcome levels (targeting bias).
    """
    if len(outcome) < 4:
        return 1.0

    y = outcome[1:-1]  # outcome_t
    x_lag = outcome[:-2].reshape(-1, 1)  # outcome_{t-1}
    x_lead = promo[2:].reshape(-1, 1)  # promo_{t+1}

    from sklearn.linear_model import LinearRegression

    # Restricted: outcome_{t-1} only
    lr_r = LinearRegression().fit(x_lag, y)
    ssr_r = float(np.sum((y - lr_r.predict(x_lag)) ** 2))

    # Unrestricted: outcome_{t-1} + promo_{t+1}
    x_u = np.column_stack([x_lag, x_lead])
    lr_u = LinearRegression().fit(x_u, y)
    ssr_u = float(np.sum((y - lr_u.predict(x_u)) ** 2))

    df_den = len(y) - 3
    if df_den <= 0 or ssr_u <= 0:
        return 1.0

    f_stat = ((ssr_r - ssr_u) / 1) / (ssr_u / df_den)
    return float(1.0 - sp_stats.f.cdf(f_stat, 1, df_den))


def diagnose_reverse_causality(
    df: pd.DataFrame,
    columns: ColumnSpec,
    max_lag: int = 3,
    alpha: float = 0.05,
) -> list[ReverseCausalityResult]:
    """Run reverse causality diagnostics for each promo variable.

    Pools data across customers for the Granger and lead tests (panel-pooled).
    """
    results: list[ReverseCausalityResult] = []
    sorted_df = df.sort_values([columns.customer_id, columns.time_col])

    for var in columns.promo_vars:
        granger_ps: list[float] = []
        lead_ps: list[float] = []

        for _, cust_df in sorted_df.groupby(columns.customer_id):
            if len(cust_df) < max_lag + 3:
                continue
            outcome = cust_df[columns.outcome_col].values.astype(float)
            promo = cust_df[var].values.astype(float)
            granger_ps.append(_granger_test(outcome, promo, max_lag))
            lead_ps.append(_lead_test(outcome, promo))

        # Combine p-values across customers using Fisher's method
        if granger_ps:
            _, granger_combined = sp_stats.combine_pvalues(granger_ps, method="fisher")
        else:
            granger_combined = 1.0
        if lead_ps:
            _, lead_combined = sp_stats.combine_pvalues(lead_ps, method="fisher")
        else:
            lead_combined = 1.0

        flagged = lead_combined < alpha
        rec = TemporalAlignment.LAGGED if flagged else TemporalAlignment.CONTEMPORANEOUS

        results.append(
            ReverseCausalityResult(
                variable=var,
                granger_p_value=float(granger_combined),
                lead_test_p_value=float(lead_combined),
                flagged=flagged,
                recommendation=rec,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Panel balance and quality
# ---------------------------------------------------------------------------
def diagnose_panel(
    df: pd.DataFrame,
    columns: ColumnSpec,
) -> PanelDiagnostic:
    """Check panel balance, zero-variance columns, and multicollinearity."""
    customers = df[columns.customer_id].unique()
    periods = df[columns.time_col].unique()
    n_cust = len(customers)
    n_periods = len(periods)
    expected = n_cust * n_periods
    actual = len(df)

    # Per-customer completeness
    counts = df.groupby(columns.customer_id)[columns.time_col].nunique()
    complete = int((counts == n_periods).sum())

    # Zero-variance columns (within-customer) — skip categorical columns
    feature_cols = columns.all_feature_cols()
    numeric_feature_cols = [c for c in feature_cols if c not in columns.categorical_vars]
    zero_var: list[str] = []
    for col in numeric_feature_cols:
        per_cust_var = df.groupby(columns.customer_id)[col].var()
        if (per_cust_var.fillna(0) == 0).mean() > 0.9:
            zero_var.append(col)

    # VIF (report only — does not fail)
    high_vif: list[tuple[str, float]] = []
    numeric_features = [c for c in feature_cols if c not in columns.categorical_vars]
    if len(numeric_features) >= 2:
        try:
            from numpy.linalg import LinAlgError

            X = df[numeric_features].dropna().values.astype(float)
            if X.shape[0] > X.shape[1] + 1:
                corr = np.corrcoef(X, rowvar=False)
                try:
                    inv_corr = np.linalg.inv(corr)
                    vifs = np.diag(inv_corr)
                    for i, col in enumerate(numeric_features):
                        if vifs[i] > 5.0:
                            high_vif.append((col, float(vifs[i])))
                except (np.linalg.LinAlgError, LinAlgError):
                    pass
        except Exception:
            pass

    return PanelDiagnostic(
        n_customers=n_cust,
        n_periods=n_periods,
        expected_rows=expected,
        actual_rows=actual,
        missing_rows=expected - actual,
        customers_complete=complete,
        customers_incomplete=n_cust - complete,
        zero_variance_cols=zero_var,
        high_vif_cols=high_vif,
    )


def balance_panel(
    df: pd.DataFrame,
    columns: ColumnSpec,
) -> pd.DataFrame:
    """Explode panel to full customer × period grid, filling missing rows."""
    customers = df[columns.customer_id].unique()
    periods = sorted(df[columns.time_col].unique())
    idx = pd.MultiIndex.from_product(
        [customers, periods], names=[columns.customer_id, columns.time_col]
    )
    balanced = df.set_index([columns.customer_id, columns.time_col]).reindex(idx)

    # Fill outcome and promo with 0 (no activity in missing periods)
    fill_zero = [columns.outcome_col] + columns.promo_vars
    for col in fill_zero:
        if col in balanced.columns:
            balanced[col] = balanced[col].fillna(0)

    # Forward-fill time-invariant columns per customer
    invariant = columns.categorical_vars + ([columns.geo_var] if columns.geo_var else [])
    for col in invariant:
        if col in balanced.columns:
            balanced[col] = balanced.groupby(level=0)[col].ffill().bfill()

    # Fill remaining controls with 0
    for col in columns.control_vars:
        if col in balanced.columns:
            balanced[col] = balanced[col].fillna(0)

    return balanced.reset_index()


# ---------------------------------------------------------------------------
# Feature engineering: adstock and lags
# ---------------------------------------------------------------------------
def apply_geometric_adstock(
    series: pd.Series,
    decay: float,
    include_contemporaneous: bool = True,
) -> pd.Series:
    """Apply geometric adstock transformation to a series.

    adstock_t = x_t + decay * adstock_{t-1}  (if include_contemporaneous)
    adstock_t = decay * (x_{t-1} + decay * adstock_{t-2})  (if not)
    """
    vals = series.values.astype(float)
    out = np.zeros_like(vals)
    if include_contemporaneous:
        out[0] = vals[0]
        for t in range(1, len(vals)):
            out[t] = vals[t] + decay * out[t - 1]
    else:
        for t in range(1, len(vals)):
            out[t] = decay * (vals[t - 1] + (out[t - 1] if t > 1 else 0.0))
    return pd.Series(out, index=series.index, name=series.name)


def add_lags(
    df: pd.DataFrame,
    columns: ColumnSpec,
    max_lag: int,
) -> pd.DataFrame:
    """Add lagged versions of promo variables per customer."""
    df = df.copy()
    df = df.sort_values([columns.customer_id, columns.time_col])
    for var in columns.promo_vars:
        for lag in range(1, max_lag + 1):
            col_name = f"{var}_lag{lag}"
            df[col_name] = df.groupby(columns.customer_id)[var].shift(lag).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Main preparation entry point
# ---------------------------------------------------------------------------
def prepare_data(
    source: pd.DataFrame | str | Path,
    config: RunConfig,
) -> PreparedData:
    """Full data preparation pipeline: load → validate → balance → diagnose → engineer."""
    df = load_data(source)

    # Validate column spec
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")

    missing_cols = [
        c for c in [config.columns.customer_id, config.columns.time_col, config.columns.outcome_col]
        + config.columns.promo_vars + config.columns.control_vars
        + ([config.columns.geo_var] if config.columns.geo_var else [])
        if c not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Panel diagnostic
    panel_diag = diagnose_panel(df, config.columns)

    # Balance panel if needed
    if panel_diag.missing_rows > 0:
        warnings.warn(
            f"Panel is unbalanced ({panel_diag.missing_rows} missing rows). "
            "Filling missing periods with zeros for outcome/promo variables.",
            stacklevel=2,
        )
        df = balance_panel(df, config.columns)
        panel_diag = diagnose_panel(df, config.columns)

    # Distribution diagnostic
    dist_diag = diagnose_distribution(df[config.columns.outcome_col])

    # Auto-detect objective if requested
    if config.objective == "auto":
        config.objective = dist_diag.recommended_objective

    # Reverse causality diagnostic
    rc_results = diagnose_reverse_causality(
        df, config.columns, max_lag=config.max_lag
    )

    # Apply recommended temporal alignment for flagged variables
    for rc in rc_results:
        if rc.flagged and rc.variable not in config.temporal_alignment:
            warnings.warn(
                f"Reverse causality detected for '{rc.variable}' "
                f"(lead test p={rc.lead_test_p_value:.4f}). "
                f"Defaulting to lagged alignment.",
                stacklevel=2,
            )
            config.temporal_alignment[rc.variable] = TemporalAlignment.LAGGED

    # Sort for temporal operations
    df = df.sort_values([config.columns.customer_id, config.columns.time_col]).reset_index(
        drop=True
    )

    # Apply carryover / temporal alignment per variable
    if config.carryover_method == config.carryover_method.LAG:
        df = add_lags(df, config.columns, config.max_lag)
    # Geometric/Weibull adstock is applied during Optuna tuning (decay is a hyperparameter)

    return PreparedData(
        df=df,
        config=config,
        panel_diagnostic=panel_diag,
        distribution_diagnostic=dist_diag,
        reverse_causality_results=rc_results,
    )
