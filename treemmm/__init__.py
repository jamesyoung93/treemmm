"""TreeMMM — Tree-based Market Mix Modeling with SHAP attribution."""

__version__ = "0.3.1"

from treemmm.core.config import ColumnSpec, Objective, RunConfig
from treemmm.pipeline import PipelineResult, run

__all__ = [
    "ColumnSpec",
    "Objective",
    "PipelineResult",
    "RunConfig",
    "__version__",
    "run",
]
