# Citation Audit

This audit covers all 54 entries in `paper/refs.bib`. The 21 entries cited by `paper/treemmm_ijf.tex` were checked against records retrieved from Crossref, the arXiv Atom API, or an authoritative publisher/project page. The eight specifically identified dormant entries were checked and repaired as well. The remaining 25 entries are uncited and were not retrieved during this bounded, cited-first pass; they remain unchanged and are explicitly flagged for author review.

## Summary

| Measure | Count |
|---|---:|
| Bibliography entries | 54 |
| Cited entries | 21 |
| Cited entries closed | 21 |
| Cited entries VERIFIED | 11 |
| Cited entries FIXED | 10 |
| Dormant entries FIXED | 8 |
| Overall VERIFIED | 11 |
| Overall FIXED | 18 |
| NEEDS AUTHOR REVIEW | 25 |

**Cited-reference closure:** 21/21 cited entries were matched to authoritative records; no cited entry remains under `NEEDS AUTHOR REVIEW`.

## Entry-level audit

| Bibkey | Retrieved authoritative URL(s) | Method | Status |
|---|---|---|---|
| `lundberg2017unified` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | Official publisher proceedings record; matched title, authors, year, venue, volume, and pages | VERIFIED |
| `lundberg2020trees` | [Crossref API](https://api.crossref.org/works/10.1038/s42256-019-0138-9) | DOI exact-record lookup; identifier resolved to the cited Nature Machine Intelligence article | VERIFIED |
| `mulc2025nnn` | [arXiv Atom API](https://export.arxiv.org/api/query?id_list=2504.06212) | arXiv identifier lookup; matched title, author list, year, and manuscript | VERIFIED |
| `gong2024causalmmm` | [Crossref API](https://api.crossref.org/works/10.1145/3616855.3635766) | DOI exact-record lookup; matched the WSDM paper and recovered its page range | FIXED (missing pages 238--246 and arXiv URL -> pages 238--246 and canonical DOI URL) |
| `romano2019conformalized` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `heskes2020causal` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=2011.01625) | Official proceedings record plus arXiv identifier lookup; matched title, authors, year, venue, pages, and preprint | FIXED (arXiv DOI recorded as proceedings DOI, missing pages, and arXiv landing URL -> no proceedings DOI, pages 4778--4789, official NeurIPS URL, and arXiv eprint) |
| `janzing2020feature` | [PMLR proceedings](https://proceedings.mlr.press/v108/janzing20a.html); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=1910.13413) | Official PMLR record plus arXiv identifier lookup; matched title, authors, year, venue, volume, pages, and preprint | FIXED (arXiv DOI recorded as proceedings DOI and abbreviated venue metadata -> no proceedings DOI, canonical PMLR venue/series, official PMLR URL, and arXiv eprint) |
| `athey2019generalized` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `wager2018estimation` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `jin2017bayesian` | [Google Research](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/) | Official institutional publication record; matched title, authors, year, and report provenance | FIXED (institution Google -> Google Inc.) |
| `rozenfeld2024causal` | [arXiv Atom API](https://export.arxiv.org/api/query?id_list=2409.06157) | arXiv identifier lookup; matched title, author, year, and manuscript | VERIFIED |
| `amoukou2022accurate` | [PMLR proceedings](https://proceedings.mlr.press/v151/amoukou22a.html); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=2106.03820) | Official PMLR record plus arXiv identifier lookup; matched title, full author list, year, venue, volume, pages, and preprint | FIXED (authors Amoukou and Brunel and unrelated arXiv 2202.01483 -> authors Amoukou, Salaun, and Brunel and arXiv 2106.03820; added pages 2448--2465 and canonical PMLR metadata) |
| `sundararajan2020many` | [PMLR proceedings](https://proceedings.mlr.press/v119/sundararajan20b.html); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=1908.08474) | Official PMLR record plus arXiv identifier lookup; matched title, authors, year, venue, volume, pages, and the correct preprint | FIXED (unrelated arXiv 1911.11718 -> arXiv 1908.08474; replaced arXiv-as-proceedings DOI with canonical PMLR metadata) |
| `tirumala2025deepcausalmmm` | [Crossref API](https://api.crossref.org/works/10.21105/joss.09914) | DOI exact-record lookup; identifier resolved to the JOSS article and matched title, author, publication year, volume, issue, and article number | VERIFIED |
| `dew2024mmm` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `duan1983smearing` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `runge2024robyn` | [arXiv Atom API](https://export.arxiv.org/api/query?id_list=2403.14674) | arXiv identifier lookup; matched title, authors, year, and manuscript | VERIFIED |
| `ke2017lightgbm` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Official publisher proceedings record; matched title, authors, year, venue, volume, and pages | FIXED (journal-style entry without pages or publisher -> proceedings entry with pages 3146--3154 and Curran Associates, Inc.) |
| `chen2016xgboost` | [Crossref API](https://api.crossref.org/works/10.1145/2939672.2939785) | DOI exact-record lookup; identifier resolved to the ACM SIGKDD paper and matched title, authors, year, venue, and pages | VERIFIED |
| `prokhorenkova2018catboost` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) | Official publisher proceedings record; matched title, authors, year, venue, volume, and pages | VERIFIED |
| `akiba2019optuna` | [Crossref API](https://api.crossref.org/works/10.1145/3292500.3330701); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=1907.10902) | DOI exact-record lookup plus arXiv identifier lookup; matched the archival KDD paper and its preprint | FIXED (arXiv-only article with arXiv DOI -> ACM KDD proceedings entry with DOI 10.1145/3292500.3330701 and pages 2623--2631; retained arXiv 1907.10902 as eprint) |
| `salvatier2016pymc3` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `meridian2024google` | [official CITATION.cff](https://github.com/google/meridian/blob/main/CITATION.cff) | Official project citation metadata; matched the software repository and current citation identity | FIXED (Google; Meridian: A Flexible Bayesian Marketing Mix Modeling Framework; 2024 -> Google Meridian Marketing Mix Modeling Team; Meridian: Marketing Mix Modeling; 2026) |
| `pymcmarketing2023` | [PyPI release metadata](https://pypi.org/pypi/pymc-marketing/0.19.4/json); [GitHub v0.19.4 tag](https://github.com/pymc-labs/pymc-marketing/tree/0.19.4) | Official package metadata and version-pinned project tag; matched package name, owner, version, and release date | FIXED (expanded project subtitle, year 2023, and floating repository URL -> PyMC-Marketing, year 2026, and version-pinned 0.19.4 URL) |
| `orbit2021uber` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `sun2017bayesian` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `bergmeir2012cv` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `hyndman2021fpp` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `makridakis2020m4` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `makridakis2022m5` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `bandara2020forecasting` | [Crossref API](https://api.crossref.org/works/10.1016/j.eswa.2019.112896) | DOI exact-record lookup; identifier resolved to the Expert Systems with Applications article and matched authors, year, venue, volume, and article number | FIXED (subtitle A Comparative Study -> A clustering approach) |
| `januschowski2020criteria` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `frye2020asymmetric` | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d770c496aa3da6d2c3f2bd19e7b9d6b-Abstract.html); [arXiv Atom API](https://export.arxiv.org/api/query?id_list=1910.06358) | Official proceedings record plus arXiv identifier lookup; matched title, authors, year, venue, pages, and the correct preprint | FIXED (unrelated arXiv 1912.12837 -> arXiv 1910.06358; added pages 1229--1239 and official NeurIPS metadata) |
| `aas2021explaining` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `shapley1953value` | [Crossref API](https://api.crossref.org/works/10.1515/9781400881970-018) | DOI exact-record lookup and DOI resolution; matched the Princeton chapter, editors, year, book, and canonical page range | FIXED (dead Princeton URL and pages 307--317 -> DOI 10.1515/9781400881970-018, stable DOI URL, and pages 307--318) |
| `athey2017state` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `chernozhukov2018double` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `kunzel2019metalearners` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `pearl2009causality` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `broadbent1979oneway` | [Market Research Society digitized archive](https://www.mrs.org.uk/blog/ijmr/digitisation-of-mrs-journal-makes-papers-archive-from-19591990-available-) | Official society archive issue scan; title/author and first and final pages confirmed in volume 21, issue 3 | VERIFIED |
| `little1979aggregate` | [Crossref API](https://api.crossref.org/works/10.1287/opre.27.4.629) | DOI exact-record lookup; identifier resolved to the Operations Research article and matched title, author, year, venue, volume, issue, and pages | FIXED (title Aggregate Advertising Models: The State of the Art -> Feature Article---Aggregate Advertising Models: The State of the Art) |
| `tellis2006modeling` | [Crossref API](https://api.crossref.org/works/10.4135/9781412973380.n24) | DOI exact-record lookup and SAGE DOI resolution; matched the chapter, handbook, editors, year, publisher, and pages | FIXED (unrelated Springer chapter DOI 10.1007/978-0-306-48382-3_3, book, and pages 51--84 -> SAGE DOI 10.4135/9781412973380.n24, The Handbook of Marketing Research, and pages 506--522) |
| `dekimpe1995persistence` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `hanssens2003market` | [Crossref API](https://api.crossref.org/works/10.1007/b109775); [Springer DOI landing](https://doi.org/10.1007/b109775) | DOI exact-record lookup plus publisher resolution; matched the second-edition book and its softcover ISBN/year | FIXED (unrelated DOI 10.1007/978-1-4615-4675-7 -> DOI 10.1007/b109775; added series volume and ISBN 978-1-4020-7368-7) |
| `naik2003synergy` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `leone1995generalizing` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `naik1998planning` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `hanssens2016pauwels` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `chan2017challenges` | [Google Research landing](https://research.google/pubs/challenges-and-opportunities-in-media-mix-modeling/); [official report PDF](https://research.google.com/pubs/archive/45998.pdf) | Official institutional landing page and report PDF; formal PDF matched title, David Chan, Michael Perry, year, and Google provenance | VERIFIED |
| `manchanda2004response` | [Crossref API](https://api.crossref.org/works/10.1509/jmkr.41.4.467.47005) | DOI exact-record lookup; identifier resolved to the Journal of Marketing Research article and matched title, authors, year, volume, issue, and pages | VERIFIED |
| `hyndman2006another` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `mccullagh1989glm` | — | Not retrieved (uncited; outside priority scope) | NEEDS AUTHOR REVIEW |
| `brooks2017glmmtmb` | [Crossref API](https://api.crossref.org/works/10.32614/RJ-2017-066) | DOI exact-record lookup; identifier resolved to The R Journal article and matched authors, year, volume, issue, and pages | FIXED (title ending Zero-inflated Data and journal R Journal -> title ending Zero-inflated Generalized Linear Mixed Modeling and journal The R Journal; added DOI URL) |
| `jorgensen1987exponential` | [Crossref API](https://api.crossref.org/works/10.1111/j.2517-6161.1987.tb01685.x) | DOI exact-record lookup; identifier resolved to the JRSS Series B article and matched title, author, year, volume, issue, and pages | FIXED (journal name without Methodological and no identifier -> historical journal name with Methodological and DOI 10.1111/j.2517-6161.1987.tb01685.x) |

## Needs author review

The 25 `NEEDS AUTHOR REVIEW` records above are all defined but uncited. They were deliberately left unchanged because no authoritative record was retrieved for them during this bounded pass. No metadata claim is made for those entries here.
