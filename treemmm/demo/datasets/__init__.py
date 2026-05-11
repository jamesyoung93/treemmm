"""Demo datasets for TreeMMM benchmarking.

Usage:
    from treemmm.demo.datasets import generate_pharma_dataset
    ds = generate_pharma_dataset(n_customers=500, n_periods=24)
"""
from treemmm.demo.datasets.cpg_brand import generate_cpg_dataset
from treemmm.demo.datasets.linear_baseline import generate_linear_dataset
from treemmm.demo.datasets.pharma_brand import generate_pharma_dataset
from treemmm.demo.datasets.saas_brand import generate_saas_dataset

__all__ = [
    "generate_pharma_dataset",
    "generate_cpg_dataset",
    "generate_saas_dataset",
    "generate_linear_dataset",
]
