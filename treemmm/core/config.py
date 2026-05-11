"""RunConfig — central configuration for a TreeMMM pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Objective(str, Enum):
    """Supported GBT objective functions mapped to outcome distributions."""

    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    TWEEDIE = "tweedie"
    GAMMA = "gamma"

    @property
    def link(self) -> Literal["identity", "log"]:
        """Return the link function implied by this objective."""
        if self is Objective.GAUSSIAN:
            return "identity"
        return "log"

    @property
    def lgbm_objective(self) -> str:
        """Map to LightGBM objective parameter value."""
        return {
            Objective.GAUSSIAN: "regression",
            Objective.POISSON: "poisson",
            Objective.TWEEDIE: "tweedie",
            Objective.GAMMA: "gamma",
        }[self]

    @property
    def lgbm_metric(self) -> str:
        """Map to LightGBM eval metric for Optuna."""
        return {
            Objective.GAUSSIAN: "rmse",
            Objective.POISSON: "poisson",
            Objective.TWEEDIE: "tweedie",
            Objective.GAMMA: "gamma",
        }[self]


class CarryoverMethod(str, Enum):
    """Adstock / carryover transformation method."""

    GEOMETRIC = "geometric"
    WEIBULL = "weibull"
    LAG = "lag"


class TemporalAlignment(str, Enum):
    """How a promotional variable aligns temporally with the outcome."""

    CONTEMPORANEOUS = "contemporaneous"
    LAGGED = "lagged"
    ADSTOCK_ONLY = "adstock_only"


class BacktestStrategy(str, Enum):
    """Time-series cross-validation strategy."""

    ROLLING_ORIGIN = "rolling_origin"
    PERIOD_JUMP = "period_jump"


@dataclass
class ColumnSpec:
    """Declares which DataFrame columns play which role."""

    customer_id: str
    time_col: str
    outcome_col: str
    promo_vars: list[str]
    control_vars: list[str] = field(default_factory=list)
    geo_var: str | None = None
    categorical_vars: list[str] = field(default_factory=list)

    def all_feature_cols(self) -> list[str]:
        """Return all feature columns (promo + control + categorical)."""
        return self.promo_vars + self.control_vars + self.categorical_vars

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors: list[str] = []
        if not self.promo_vars:
            errors.append("At least one promotional variable is required.")
        overlap = set(self.promo_vars) & set(self.control_vars)
        if overlap:
            errors.append(f"Variables appear in both promo and control: {overlap}")
        cat_promo = set(self.categorical_vars) & set(self.promo_vars)
        if cat_promo:
            errors.append(f"Variables appear in both categorical and promo: {cat_promo}")
        cat_ctrl = set(self.categorical_vars) & set(self.control_vars)
        if cat_ctrl:
            errors.append(f"Variables appear in both categorical and control: {cat_ctrl}")
        all_cols = [self.customer_id, self.time_col, self.outcome_col]
        all_cols += self.promo_vars + self.control_vars + self.categorical_vars
        if self.geo_var:
            all_cols.append(self.geo_var)
        dupes = [c for c in all_cols if all_cols.count(c) > 1]
        if dupes:
            errors.append(f"Duplicate column assignments: {set(dupes)}")
        return errors


@dataclass
class RunConfig:
    """Full configuration for a TreeMMM pipeline run.

    Attributes:
        columns: Column role declarations.
        objective: Outcome distribution / GBT objective. Use 'auto' sentinel
            to let the data handler recommend based on outcome characteristics.
        backtest: Time-series CV strategy.
        min_train_frac: Minimum fraction of time periods used for training
            in each CV fold.
        n_optuna_trials: Bayesian optimization budget per fold.
        carryover_method: Adstock transformation method.
        temporal_alignment: Per-variable temporal alignment specification.
            Keys are promo_var names; unlisted variables default to
            'contemporaneous'. Variables flagged by reverse causality
            diagnostic should be set to 'lagged'.
        tweedie_variance_power: Tweedie power parameter (1 < p < 2).
            Only used when objective is TWEEDIE.
        max_lag: Maximum lag periods for LAG carryover or Granger test.
        random_state: Reproducibility seed.
        adstock_decay: Geometric adstock decay rate(s) to apply to promo
            features BEFORE model fitting.  ``None`` means no adstock
            preprocessing.  A single ``float`` applies the same rate to
            all promo channels.  A ``dict[str, float]`` maps each channel
            to its own rate; channels not present in the dict are left
            untransformed (decay=0).  Valid values are in ``[0, 1)``.
    """

    columns: ColumnSpec
    objective: Objective | Literal["auto"] = "auto"
    backtest: BacktestStrategy = BacktestStrategy.ROLLING_ORIGIN
    min_train_frac: float = 0.6
    n_optuna_trials: int = 50
    carryover_method: CarryoverMethod = CarryoverMethod.GEOMETRIC
    temporal_alignment: dict[str, TemporalAlignment] = field(default_factory=dict)
    tweedie_variance_power: float = 1.5
    max_lag: int = 3
    random_state: int = 42
    adstock_decay: float | dict[str, float] | None = None

    def get_alignment(self, var: str) -> TemporalAlignment:
        """Return temporal alignment for a variable, defaulting to contemporaneous."""
        return self.temporal_alignment.get(var, TemporalAlignment.CONTEMPORANEOUS)

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty if valid)."""
        errors = self.columns.validate()
        if not 0.3 <= self.min_train_frac <= 0.9:
            errors.append(f"min_train_frac must be in [0.3, 0.9], got {self.min_train_frac}")
        if self.n_optuna_trials < 1:
            errors.append("n_optuna_trials must be >= 1")
        if self.objective == Objective.TWEEDIE:
            if not 1.0 < self.tweedie_variance_power < 2.0:
                errors.append(
                    f"tweedie_variance_power must be in (1, 2), "
                    f"got {self.tweedie_variance_power}"
                )
        bad_align = set(self.temporal_alignment.keys()) - set(self.columns.promo_vars)
        if bad_align:
            errors.append(
                f"temporal_alignment keys not in promo_vars: {bad_align}"
            )
        return errors
