# arxiv Submission Metadata

## Title

TreeMMM: Tree-Based Marketing Mix Modeling with SHAP Attribution and Automatic Interaction Discovery

## Authors

James Young

## Abstract (1,464 characters — within arxiv 1,920-character limit)

Marketing mix modeling (MMM) at the healthcare-professional (HCP) panel
scale presents a regime fundamentally different from the aggregate
weekly time-series that motivated classical Bayesian MMM. We present
TreeMMM, a Python package pairing LightGBM with SHAP for
distribution-aware channel attribution, automatic interaction discovery,
and marginal-ROI optimization on panel data. We benchmark TreeMMM
against five baselines (GLMM-Naive, GLMM-Oracle, PyMC-Hier-Naive,
PyMC-Hier-Oracle, PyMC-Marketing) on four synthetic datasets. On the
three non-linear datasets, TreeMMM achieves attribution share-MAPE of
17.9% +/- 0.2% vs. 22.2% +/- 0.3% for GLMM-Naive (gap 4.3 pp +/-
0.4 pp, N=5 seeds). A distributional-GLM experiment confirms the
advantage is not an artifact of the log1p link workaround: a properly
specified Poisson/Tweedie/Gamma GLM (GLMMDist-Naive) performs no
better, and on two of three datasets slightly worse. TreeMMM
automatically detects 5 of 6 planted channel interactions without
pre-specification (F1=0.56 at default thresholds); regression-based
baselines detect zero. On a geo-panel DGP designed for
PyMC-Marketing's parametric strengths, TreeMMM (29.7%) outperforms
PyMC-Marketing (52.1%) and Meridian (57.0%). A prior-variance sweep
confirms that at n=3,000 x 36 periods all channels move less than
0.1 pp across a 4x prior-width bracket. Code and data:
https://github.com/jamesyoung93/treemmm.

## Comments Line (arxiv "comments" field)

33 pages, 13 figures, 9 tables. Code and benchmark data: https://github.com/jamesyoung93/treemmm. Package v0.2.1. Submitted to International Journal of Forecasting.

## Primary Subject Class

stat.AP  (Statistics — Applications)

## Cross-list Categories

- cs.LG   (Machine Learning)
- stat.ML (Machine Learning)

## MSC Codes (optional)

- 62P30 — Statistics applied to social sciences (business analytics)
- 62M10 — Time series, auto-correlation, regression, etc.
- 68T05 — Learning and adaptive systems

## ACM Codes (optional)

- I.2.6  — Learning
- J.1    — Administrative Data Processing — Business

## License

Creative Commons Attribution 4.0 (CC BY 4.0)
(Standard choice for stat.AP papers; allows IJF to reprint with attribution)

## Submission Checklist

- [ ] arxiv account registered at https://arxiv.org/user/register
- [ ] Source files ready: `paper/treemmm_arxiv.tar.gz` (contains treemmm_ijf.tex, refs.bib, figures/)
- [ ] Upload tarball at https://arxiv.org/submit
- [ ] Select primary class: stat.AP
- [ ] Add cross-list: cs.LG, stat.ML
- [ ] Paste abstract from this file (plain text version above)
- [ ] Set "Comments" line (see above)
- [ ] Verify compiled PDF on arxiv preview (arxiv re-compiles from source)
- [ ] Mint Zenodo DOI for v0.2.1 at https://zenodo.org (create new upload, link to GitHub tag)
- [ ] Update "TO BE MINTED" placeholder in treemmm_ijf.tex Code Availability section with Zenodo DOI
- [ ] Recompile and re-tar after Zenodo DOI is known, before final submission
