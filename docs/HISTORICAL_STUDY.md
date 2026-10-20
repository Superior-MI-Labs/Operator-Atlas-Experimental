# Historical sunspot study — reproducibility and interpretation

Source: `examples/fixtures/sunspots_1700_2008.csv`. This is a local snapshot of statsmodels' public-domain annual sunspots dataset, 309 rows, 1700–2008. Original source National Geophysical Data Center. A later independent archive should verify data revisions before further publication.

Command: `./launch.sh study-real`. It reproduces `results/historical_sunspots_v1.json` without requiring statsmodels after installation. The fixture was exported from statsmodels on 2026-10-20, but only NumPy is needed to run the experiment.

Predeclared chronological cutoffs: 1880, 1900, 1920, 1940, 1960, 1980; train on all preceding years, test on next 15. Each candidate formula is selected **within prior years only**, and neither candidate nor AR2 receives future targets. For 15 future-year predictions, AR2 is recursively forecast using predicted previous values (not leaked actual test-year values). The production system refuses out-of-domain predictions; only this offline stress study deliberately evaluates them to measure failure.

The resulting experiment supports **failure of the supplied equation grammar as a dynamical solar predictor** in this setup. It does not prove mathematical organization useless in general. Try more appropriate time-series models, stateful features, uncertainty intervals and model-family selectors before proposing solar prediction applications.

![Observed mean error](../assets/sunspot_backtest.png)
