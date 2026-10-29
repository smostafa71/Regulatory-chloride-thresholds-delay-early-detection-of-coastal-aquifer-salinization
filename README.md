# Code and data: "Regulatory chloride thresholds delay early detection of coastal aquifer salinization"

Seyed Mostafa Biazar, Southwest Florida Water Management District (Journal of Hydrology, manuscript HYDROL84325).

This archive reproduces the numbers, tables and figures of the article and its Supplementary Information from public U.S. groundwater-quality data.

## Contents

| Folder or file | What it holds |
|---|---|
| `raw/raw_XX.csv` | The analysis snapshot of dissolved-chloride results for the 17 coastal states. These were retrieved from the Water Quality Portal (resultPhysChem profile; NWIS and partner organizations) in 2026. Each file is unchanged as downloaded. |
| `raw/national.parquet`, `raw/chloride_national.csv` | The national analysis dataset: 218,181 results from 44,990 wells, 1980-01-01 to 2026-04-16 (site, date, chloride in mg/L, state). |
| `raw/wqp_station_pool.csv`, `raw/nwis_site_expanded_pool.csv`, `raw/meta_pool.csv`, `raw/*aquifer_codes.csv`, `raw/nat_aqfr.rdb` | Station metadata (coordinates, depth, aquifer codes) and the merged metadata table, including the geodesic distance to the marine shoreline. |
| `raw/wl/` | USGS groundwater-level series (parameter 72019) for the transition wells. |
| `raw/*.csv`, `raw/*.npy` (other) | Derived tables behind each figure and table, for example transition events, stable controls, the threshold grid, forecasts, the power analysis and the watch list. |
| `numbers.json` | Every numerical value reported in the revised article, grouped by analysis. |
| `figs/` | Figures 1–7 and S1–S3 (PNG at 300 dpi and PDF), and `TableS10_transition_inventory.csv` (the per-well inventory of the 145 transition wells). |
| `gis/` | Natural Earth 1:10 m coastline, lakes and state boundaries (public domain), used for distances and maps. |
| `*.py` | Analysis and figure scripts (see run order below). `core.py` holds shared functions. |

## Requirements

Python 3.10 or later. Install the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Run order

Run all scripts from this folder. Steps 1–5 regenerate the core results from `raw/raw_XX.csv`. We checked that they reproduce the archived `national.parquet`, the transition and stable-control lists, the lead-time and false-positive matrices, and Table S1 exactly.

| Step | Script(s) | Produces |
|---|---|---|
| 1 | `build_national.py` | Raw snapshot → harmonized long table |
| 2 | `step1_national.py` | National dataset (218,181 results, 44,990 wells); well classification; 145 transition wells |
| 3 | `step2_core.py`, `fp_followup_window.py` | Transition events, 1,439 stable controls, lead-time and false-positive matrices, threshold trade-off (Table S3), state values; false-positive rate with 3-, 5- and 7-year follow-up windows (Supplementary Note 1) |
| 4 | `step3_knee.py`, `knee_loso_original.py` | Frontier on the full grid (sensitivity); elbow bootstrap on the operational grid 100–200 mg/L (90% interval 125–175) and leave-one-state-out (1.82–2.83 yr) as reported (Fig. 5, Fig. 7b, Table S4) |
| 5 | `step4_desc.py`, `step5_cf.py` | Record descriptors (Tables S9, S14); counterfactual monitoring policies (Table S1) |
| 6 | `cf_eq6.py` | Strict alarm-date counterfactual (Table S13) |
| 7 | `run_versions.py`, `step6_dupdates.py` | Data-quality and same-day variants (Table S7; writes `raw/V1.pkl`, needed by step 10) |
| 8 | `fig3_reimpl.py`, `fig3_pred.py`, `fig3_pred2.py`, `fig3_fc.py`, `fig3_alt.py`, `forecast_compare.py`, `nominal_ci.py` | Curvature (Table S8), predictability, forecast comparison against the naive forecast (Table 2), bootstrap intervals of the nominal association |
| 9 | `ews_reimpl.py`, `ews_robust.py`, `ews_cv.py`, `ews_power.py` | Early-warning indicators and power analysis (Tables S11, S12) |
| 10 | `fetch_meta.py`, `build_meta.py`, `classify_setting.py`, `table1.py`, `covariate_models.py` | Metadata, distance to the shoreline, settings and lithology, Table 1, Table S10; adjusted covariate models and partial rank correlation (Table S6) |
| 11 | `dl_wl.py`, `wl_test.py` | Water-level download and tests (Table S15) |
| 12 | `new_A.py`, `new_G_D.py` | ROC and PPV, representativeness (Table S9), watch list (Table S16) |
| 13 | `fig2_band.py`, `fig_inventory.py`, `fig_shape.py`, `fig_tradeoff.py`, `fig_maps.py`, `fig_workflow.py`, `fig_supp.py` | Figures (style in `figstyle.py`) |
| 14 | `verify_all.py` | Independent check: recomputes about 270 reported quantities from the data and per-well tables and compares each with the value in the article (writes `verify_report.csv`) |

`dl_states.py` and `fetch_meta.py` query the Water Quality Portal and NWIS services. Re-running them downloads the current portal contents, which differ from the archived snapshot because records are added and revised continually. To reproduce the article, use the archived `raw/` files.

## Notes

- **Archived outputs of the original analysis.** Files ending in `_original.csv` in `raw/` are outputs of the original pipeline, archived in June 2026: the per-well event table (`transition_events_original.csv`), the threshold curve, the false-positive table, the counterfactual policy tables, the state lead times and the water-level series. The rebuilt pipeline reproduces them. `chloride_national.csv` holds the same 218,181 results as the archived national file; only the order of same-day results within 19 wells differs.
- **Two per-well event tables.** The threshold sweep (`step2_core.py`) sorts the national table by date before splitting it by well, while the per-well event table (`transition_events_original.csv`) keeps the original within-well order. They differ only for seven wells with several results on the same day, and all reported medians are identical. Leave-one-state-out and Table S10 use the per-well event table, as in the original analysis.

- **Same-day ordering.** A few wells have several results on the same date, and the two-consecutive-result rule depends on their order. The scripts keep the order of the archived snapshot (state order, then site, then date, stable sort). Collapsing same-day results to their median is reported as a sensitivity test (Table S7).
- **Forecasts below zero.** A remaining-time forecast is set to zero when it is negative, which happens once (linear-rate forecast, latest value already above 250 mg/L); this does not change any value in Table 2.
- **Tables from interactive sessions.** A few intermediate tables were produced interactively with the functions in `core.py`. They are included as data: `transition_events_cell51.csv`, `persist_sens.csv`, `variants_cleaning.csv`, `trans_full.csv` and `watchlist_2015_geo.csv`.
- **Virginia records.** The Virginia file contains records later than the analysis snapshot. `step1_national.py` excludes Virginia results after 2025-12-31 to restore the snapshot used in the article.

## License and citation

Code: MIT License. The data are U.S. public-domain data from the U.S. Geological Survey and partner agencies, distributed by the Water Quality Portal (https://www.waterqualitydata.us). Natural Earth data are public domain.

Please cite the article and this archive (DOI to be added on publication).
