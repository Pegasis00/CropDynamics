# Notebook → App Mapping

For auditing that nothing was redesigned. Left column is the notebook
stage/function; right column is where it now lives.

| Notebook (Stage / function) | App module |
|---|---|
| GEE JS script (Sentinel-2 pull, dedup, NDVI) | `backend/gee/extractor.py` (`GEEDataExtractor`), `gee_scripts/ndvi_extraction_cleaned.js` |
| Stage B: farm dict construction | `backend/lifecycle/farm_builder.py` (`FarmBuilder`, `LocationRecord`) |
| Stage C: `adaptive_smoothing` | `backend/signal_processing/smoothing.py` |
| Stage C: `dynamic_prominence`, `estimate_noise` | `backend/signal_processing/smoothing.py` |
| Stage C: `detect_peaks` / `detect_valleys` | `backend/signal_processing/peaks.py` (`PeakDetector`, `ValleyDetector`) |
| Stage D: `find_left/right_boundary`, `growth_score`, `decline_score`, `find_best_sos/eos` | `backend/lifecycle/boundary.py` (`BoundarySelector`) |
| Stage D: `build_candidate`, `generate_candidates`, `compute_quality_metrics` | `backend/lifecycle/candidates.py` (`CandidateGenerator`) |
| Stage D: `check_ground_truth`, `check_duration`, `validate_candidate` | `backend/lifecycle/validator.py` (`CandidateValidator`) — `check_ground_truth` kept for `mode="training"`; live inference uses the new `check_query_date_coverage`, which plays the equivalent structural role for a query that has no ground-truth month |
| Stage D: `select_best_candidate` | `backend/lifecycle/selector.py` (`LifecycleSelector`) |
| Stage E: `plot_smoothing`, `plot_signal`, `plot_selected_cycle` | `backend/visualization/plots.py` (`PlotGenerator`) — same series/markers, emitted as JSON for the web frontend instead of matplotlib figures |
| Stage G: `build_cycle_features` (~71 features) | `backend/features/extractor.py` (`FeatureExtractor.extract`) — ported line-for-line, including all helper functions (`_safe_div`, `_longest_true_run`, `_first_crossing_day`, `_safe_gradient`, `safe_peak_width`, `safe_peak_prominence`) |
| Model Building: `MODEL_CONFIGS`, `make_pipeline`, `train_and_evaluate`, comparison table, artifact export | `backend/models/trainer.py` (`ModelTrainer`) — ported as a standalone offline script; **not** run by the API |
| Model Building: artifact loading | `backend/models/registry.py` (`ModelRegistry`) |
| Prediction Strategy (single selected model, one final prediction) | `backend/models/prediction_service.py` (`PredictionService`) |
| Full pipeline sequencing (16 steps in the spec) | `backend/pipelines/inference_pipeline.py` (`InferencePipeline`) |

## Deliberate, spec-called-out adaptations (not simplifications)

- **Ground-truth vs. query-date coverage.** The notebook's
  `check_ground_truth` compares a candidate's SOS/EOS span against a
  labeled ground-truth month, which only exists in training data. For a
  live query there's no such label, so `CandidateValidator` runs
  `check_query_date_coverage` instead — same structural role (does the
  candidate's span plausibly contain the reference point in time), just
  keyed off the query date instead of a label. This is exactly what the
  spec's pipeline step 10 ("ground-truth/query-date coverage") calls for.
- **Plots as JSON, not matplotlib PNGs.** The three notebook plotting
  functions defined *which* series and markers go on each chart. That
  logic is preserved in `PlotGenerator`; it just emits the same data as
  JSON so the frontend can render interactive charts instead of static
  images.
- **Training is a separate, opt-in script.** `backend/models/trainer.py`
  reproduces the notebook's 8-model GridSearchCV training run
  feature-for-feature, but the FastAPI app never imports or calls it —
  by design, since trained artifacts already exist and shouldn't be
  regenerated as a side effect of starting the API.
