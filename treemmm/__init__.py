"""TreeMMM — Tree-based Market Mix Modeling with SHAP attribution.

Market Mix Modeling that finds what you didn't think to look for.
"""

__version__ = "0.2.1"

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
