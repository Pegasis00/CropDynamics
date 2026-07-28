# Field Signal — Crop Classification App

A production wrapper around the reference notebook's NDVI phenology /
crop-classification pipeline. The lifecycle, feature engineering, and
single-model inference logic follow the notebook, while the live GEE
extractor is a point-only API version tuned for single coordinate queries.
See `NOTEBOOK_MAPPING.md` for the cell-by-cell mapping.

## What this does NOT do

This app does **not** train models. Training already happened in the
notebook, and the resulting `.joblib` files are expected to already exist
in `model_weights/` (see `model_weights/README.md`). `backend/models/trainer.py`
is included as a faithful, ready-to-run port of the notebook's training
code for whenever you need to retrain later — it's just never invoked
automatically.

## Layout

```
configs/            gee.yaml, model.yaml, pipeline.yaml, thresholds.yaml,
                     crop_rules.yaml — every tunable knob lives here, not in code
backend/
  gee/               GEEDataExtractor — Sentinel-2 point extraction
  lifecycle/         FarmBuilder, BoundarySelector, CandidateGenerator,
                     CandidateValidator, LifecycleSelector
  signal_processing/ SignalProcessor, PeakDetector, ValleyDetector
  features/          FeatureExtractor (~71 features x {raw, smooth})
  models/            ModelRegistry, PredictionService, trainer.py (offline only)
  visualization/      PlotGenerator — JSON plot payloads for the frontend
  pipelines/          InferencePipeline — orchestrates the full 16-step sequence
  api/                FastAPI routes + Pydantic schemas
  config/             YAML config loader
frontend/            React/Vite UI for querying, plotting, and reviewing results
model_weights/       put your trained .joblib files here
saved_crop_ndvi/     per-crop CSVs created when accepted predictions are saved
gee_scripts/         cleaned GEE JS extraction script (batch/offline use)
tests/               unit tests for signal processing + feature extraction
```

## Running it

```bash
pip install -r requirements.txt

# Earth Engine auth (one-time, or set GOOGLE_APPLICATION_CREDENTIALS for servers)
earthengine authenticate

# Earth Engine also needs a Google Cloud project.
# Either set configs/gee.yaml -> earth_engine_project, or:
set GOOGLE_CLOUD_PROJECT=your-project-id

# put your trained artifacts in model_weights/ (see model_weights/README.md)

uvicorn backend.main:app --reload --port 8000
```

In a second terminal, start the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open `http://127.0.0.1:5173`. The frontend talks to
`http://localhost:8000/api` by default. Override it with `VITE_API_BASE`
when running Vite if the backend is elsewhere.

## Deploying

The included `Dockerfile` is ready for a Hugging Face Docker Space. It builds
the React app with `VITE_API_BASE=/api`, serves the static frontend from
FastAPI, and runs Uvicorn on the Space port (`7860` by default).

For Hugging Face, add these Space secrets/settings before using live
prediction:

- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: full Google service-account JSON with
  Earth Engine access
- `GOOGLE_CLOUD_PROJECT`: the Earth Engine Google Cloud project, if different
  from `configs/gee.yaml`

For Vercel frontend-only deployment, deploy the `frontend/` directory and set
`VITE_API_BASE` to the hosted backend URL plus `/api`, for example
`https://<space-owner>-<space-name>.hf.space/api`.

After review, click **Save to CSV** to append accepted crop predictions as
SOS-to-EOS raw NDVI rows under
`saved_crop_ndvi/<crop>/<crop>.csv` with `CROP_LABEL`, `CROP_ID`, `NDVI`,
`DATE`, `QUERY_DATE`, `LATITUDE`, and `LONGITUDE`.
Use **Change prediction** before saving when the model label needs to be
corrected for data collection or later retraining.
Smooth data is the default model input. Use **Test raw data** to rerun the
current lifecycle through `model_raw_LightGBM.joblib` when LightGBM is the
selected algorithm. Raw mode detects peaks, SOS, and EOS on the raw NDVI
series; smooth mode uses the smoothed NDVI series.
Flat or low-vegetation lifecycles are rejected before model inference as
`No plant found` using `configs/thresholds.yaml`'s
`crop_signal_validation` settings.

## Configuration

Everything called out in the spec as "must be config-driven" lives in
`configs/*.yaml` — cloud threshold, extraction window, point pixel scale,
smoothing window rule, peak/valley prominence factor, `SIG_MULTIPLIER`,
per-crop duration bounds, and which of the 8 trained models is selected for
production. Change the YAML, restart the app — no code edits needed.

`configs/crop_rules.yaml` holds second-stage post-model verification rules.
For example, Sugarcane is only accepted when the extracted lifecycle duration
is greater than 280 days; otherwise the final label becomes `Other Crop`.

## Crop classes

`Cotton`, `Onion`, `Paddy`, `Sugarcane`, `Wheat` — read from
`configs/model.yaml`'s `classes` list. If your training data's crop set
changes, edit that list (and the matching `duration_stats` entries in
`configs/thresholds.yaml`); nothing in the code assumes a fixed crop count.

## Edge cases

| Situation | Response |
|---|---|
| Too few NDVI observations | `"Not enough NDVI observations."` (422) |
| Point doesn't look like cropland | `"Not agricultural land"` with NDVI plots (200) |
| No lifecycle passes validation | `"No valid crop lifecycle"` with NDVI plots (200) |
| Confident prediction below the trained classes | `"Other Crop"` (200, `is_other_crop: true`) |
