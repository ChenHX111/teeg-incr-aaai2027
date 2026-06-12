# TEEG-INCR: Millisecond-Scale Incremental Updates for Public-Transit Earliest-Arrival Routing at City Scale

This is the anonymous code companion for the AAAI 2027 submission of the same name.

## Layout

```
Route_Planner/    # all 13 routing-method packages (CSA, RAPTOR, ULTRA, HL, CH,
                  #  TEEG-ALT, TEEG-ALT-v2, TEEG-INCR, MG-Dial, MG-Dial-v2, ...)
Experiments/      # benchmark scripts (compiled bench, Cycle 10, walkcap, oracle)
paper/            # main.tex + supplementary.tex + bib + style files
reference/        # SHA-256 manifest of the GTFS data bundles (data not shipped)
```

## How to reproduce

1. Install: `pip install -r requirements.txt` (numpy 2.4, pandas 3.0, scipy 1.17,
   networkx 3.6, numba 0.65.1, llvmlite 0.47.0).
2. GTFS data: TfL (London), VBB (Berlin), MTA (NYC), all anchored to weekday
   2025-01-03 07:00-20:00. SHA-256 of each pre-processed bundle is in
   `reference/data_manifest.json`. Source GTFS feeds are publicly available
   from the listed agencies.
3. Static bench (Table 1): `python -m Experiments.benchmark_compiled_only --queries Q.csv --out O.csv`
4. v2 columns: `python -m Experiments.benchmark_compiled_v2 --queries Q.csv --out O.csv`
5. Cycle 10 dynamic-update matrix: `python -m Experiments.cycle10_run_city --city CITY --queries Q.csv`
6. Cycle 10 rebuild-oracle correctness: `python -m Experiments.cycle10_rebuild_oracle_city --city CITY ...`
7. Lazy-CSA-resort baseline (Table tab:lazy-csa): `python -m Experiments.cycle10_lazy_csa --city CITY ...`
8. Walk-cap sensitivity: `python -m Experiments.benchmark_walkcap_sweep --queries Q.csv --caps 5,10,15,20,30`
9. Multi-rep cross-machine: `python -m Experiments.multi_rep_fullldn --queries Q.csv --reps 3`

## License

MIT (see LICENSE).
