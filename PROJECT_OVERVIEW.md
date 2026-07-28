# Project Overview: Field Signal Crop Classification App

## 1. What this project is

This project is a crop-classification application built around NDVI time-series signals from Google Earth Engine. A user enters a latitude, longitude, and query date. The backend extracts Sentinel-2 NDVI observations around that point, detects the most relevant crop lifecycle, generates crop phenology features, runs a trained machine-learning model, and returns a crop prediction with lifecycle charts.

The project is mainly a production-style wrapper around the original notebook:

- `NDVI_CROP_CLASSIFICATION_(2) (1).ipynb`
- `NOTEBOOK_MAPPING.md`

The notebook logic has been split into backend modules, configuration files, model artifacts, a FastAPI API, and a React frontend.

## 2. What is already done here

The following work is already implemented:

- FastAPI backend for crop prediction.
- Google Earth Engine point-based NDVI extraction.
- NDVI smoothing with adaptive Savitzky-Golay filtering.
- Peak and valley detection.
- SOS, peak, and EOS lifecycle boundary selection.
- Candidate lifecycle generation, scoring, validation, and best-cycle selection.
- Feature extraction from raw and smoothed NDVI curves.
- Loading trained `.joblib` model artifacts from `model_weights/`.
- Prediction using the configured model, currently LightGBM by default.
- Confidence thresholding so low-confidence predictions become `Other Crop`.
- Post-model crop rules, such as Sugarcane needing a long enough lifecycle.
- React/Vite frontend for running predictions and viewing charts.
- Chart.js visualizations for raw/smoothed NDVI, detected markers, and selected lifecycle.
- Manual lifecycle editing from the UI.
- Raw vs smooth model testing from the UI.
- Crop label correction before saving.
- CSV export of accepted/reviewed crop lifecycles into `saved_crop_ndvi/`.
- Unit tests for signal processing, feature extraction, crop rules, and CSV writing.

## 3. Main user flow

1. User opens the frontend.
2. User enters coordinates and a query date.
3. Frontend calls the backend `/api/predict` endpoint.
4. Backend fetches Sentinel-2 NDVI observations from Google Earth Engine.
5. Backend smooths the signal and detects peaks/valleys.
6. Backend finds possible crop lifecycles around the query date.
7. Backend selects the best valid lifecycle.
8. Backend extracts raw and smooth feature sets.
9. Backend runs the configured trained model.
10. Backend applies confidence and crop-rule checks.
11. Backend returns the crop prediction, confidence, lifecycle dates, and plot payloads.
12. Frontend renders the result and charts.
13. User can edit lifecycle boundaries, switch raw/smooth mode, correct the crop label, and save accepted lifecycle data to CSV.

## 4. Backend structure

The backend is under `backend/`.

### API layer

- `backend/main.py`
  - Creates the FastAPI app.
  - Loads settings.
  - Initializes Earth Engine extraction, model registry, CSV writer, and inference pipeline.

- `backend/api/routes.py`
  - Defines API endpoints:
    - `GET /api/health`
    - `GET /api/crops`
    - `POST /api/predict`
    - `POST /api/predict/manual-lifecycle`
    - `POST /api/predictions/save`

- `backend/api/schemas.py`
  - Defines Pydantic request/response models.
  - Supports either separate latitude/longitude fields or a pasted coordinate pair.
  - Validates dates, coordinate ranges, and lifecycle input lengths.

### GEE extraction

- `backend/gee/extractor.py`
  - Extracts NDVI from `COPERNICUS/S2_SR_HARMONIZED`.
  - Uses B8 and B4 bands to compute NDVI.
  - Filters by query-date window and cloud threshold.
  - Samples the exact coordinate point at Sentinel-2 resolution.

- `gee_scripts/ndvi_extraction_cleaned.js`
  - Cleaned Google Earth Engine JavaScript script for batch/offline use.

### Lifecycle and signal processing

- `backend/signal_processing/smoothing.py`
  - Adaptive Savitzky-Golay smoothing.
  - Noise estimation.
  - Dynamic prominence logic.

- `backend/signal_processing/peaks.py`
  - Peak detection.
  - Valley detection.

- `backend/lifecycle/farm_builder.py`
  - Builds a structured location record from raw NDVI observations.

- `backend/lifecycle/boundary.py`
  - Finds SOS and EOS boundaries around peaks.
  - Uses growth and decline scoring.

- `backend/lifecycle/candidates.py`
  - Builds lifecycle candidates.
  - Computes quality metrics such as duration, amplitude, baseline, and signal-to-noise.

- `backend/lifecycle/validator.py`
  - Validates lifecycles using configured thresholds.
  - In live inference, checks whether the query date is covered by the candidate lifecycle.

- `backend/lifecycle/selector.py`
  - Chooses the best valid lifecycle candidate.

### Feature extraction and models

- `backend/features/extractor.py`
  - Extracts the notebook-style crop lifecycle features.
  - Works on both raw and smoothed NDVI curves.

- `backend/models/registry.py`
  - Loads model artifacts and the label encoder from `model_weights/`.

- `backend/models/prediction_service.py`
  - Runs the selected model.
  - Converts class probabilities into the final prediction response.
  - Applies the `Other Crop` fallback when confidence is too low.

- `backend/models/trainer.py`
  - Offline training port from the notebook.
  - Not used by the API at runtime.
  - Can be used later if the model needs retraining.

### Pipeline orchestration

- `backend/pipelines/inference_pipeline.py`
  - The main orchestrator.
  - Runs the full request lifecycle from NDVI extraction to final response.
  - Also supports manual lifecycle reruns and CSV saving.

### Visualization and export

- `backend/visualization/plots.py`
  - Builds JSON plot payloads for frontend charts.
  - Replaces notebook matplotlib output with browser-rendered chart data.

- `backend/exporters/crop_ndvi_writer.py`
  - Saves accepted crop lifecycle NDVI points to per-crop CSV files.
  - Output location is `saved_crop_ndvi/<crop>/<crop>.csv`.

## 5. Frontend structure

The frontend is under `frontend/` and uses React, Vite, and Chart.js.

- `frontend/src/App.jsx`
  - Main UI.
  - Handles coordinate/date input.
  - Calls backend prediction endpoints.
  - Renders model result, confidence, lifecycle metrics, and charts.
  - Supports lifecycle editing, raw/smooth reruns, crop label correction, and CSV saving.

- `frontend/src/styles.css`
  - Application styling.

- `frontend/package.json`
  - Frontend scripts:
    - `npm run dev`
    - `npm run build`
    - `npm run preview`

The frontend talks to `http://localhost:8000/api` by default. This can be changed with `VITE_API_BASE`.

## 6. Configuration

Most important behavior is controlled by YAML files in `configs/`.

- `configs/gee.yaml`
  - Earth Engine collection.
  - Cloud threshold.
  - NDVI extraction window.
  - Pixel scale.
  - Minimum observation count.
  - Earth Engine project.

- `configs/pipeline.yaml`
  - Smoothing window rules.
  - Polynomial order.
  - Peak/valley detection thresholds.
  - Boundary scoring windows.
  - Candidate quality settings.

- `configs/model.yaml`
  - Selected algorithm.
  - Selected feature set.
  - Model artifact naming.
  - Crop classes.
  - Unknown-crop confidence floor.

- `configs/thresholds.yaml`
  - Crop lifecycle duration statistics.
  - Location validation thresholds.
  - Crop signal validation thresholds.

- `configs/crop_rules.yaml`
  - Post-model verification rules.
  - Example: Sugarcane must have duration greater than 280 days.
  - Example: Wheat must have duration less than 180 days.

This is useful because algorithm settings can be changed without editing Python code.

## 7. Model artifacts and data files

- `model_weights/`
  - Contains trained model artifacts.
  - Current available artifacts include:
    - `model_smooth_LightGBM.joblib`
    - `model_raw_LightGBM.joblib`
    - `label_encoder.joblib`

- `combined_ndvi.csv`
  - Combined NDVI dataset used by the project/notebook workflow.

- `saved_crop_ndvi/`
  - Stores accepted lifecycle exports by crop.
  - Current folders include Cotton, Onion, Paddy, Sugarcane, and Wheat.

The API does not train models when it starts. It expects trained model files to already exist in `model_weights/`.

## 8. Prediction outputs

A normal prediction response includes:

- `predicted_crop`
- `confidence`
- `is_other_crop`
- `feature_set`
- `lifecycle`
  - `sos_date`
  - `peak_date`
  - `eos_date`
  - `duration_days`
- `plots`
  - raw vs smoothed NDVI
  - peaks and valleys
  - selected lifecycle
- `saved_crop_id`

Possible non-standard prediction labels include:

- `Not agricultural land`
- `No valid crop lifecycle`
- `No plant found`
- `Other Crop`

## 9. Tests

Tests are under `tests/`.

Current test areas include:

- Signal smoothing and processing.
- Feature extraction.
- Prediction save behavior.
- Crop signal validation.
- Crop rules.
- Crop NDVI CSV writing.

Run tests with:

```bash
pytest
```

## 10. How to run locally

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Earth Engine if needed:

```bash
earthengine authenticate
```

Start the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## 11. Important limitations

- The app depends on Google Earth Engine access.
- The API requires model artifacts to be present in `model_weights/`.
- Runtime training is intentionally not part of the API.
- Prediction quality depends on the original notebook training data and the available NDVI observations for the selected point/date.
- If too few NDVI observations are available, the API returns a validation error instead of forcing a prediction.
- If the signal is flat, weak, non-agricultural, or outside crop rules, the API returns a fallback label rather than forcing one of the trained crop classes.

## 12. Short summary

In short, this project turns a notebook-based NDVI crop-classification workflow into a usable web app. The backend performs Earth Engine NDVI extraction, signal processing, lifecycle detection, feature engineering, model inference, rule checks, and CSV export. The frontend provides a review workspace where users can inspect charts, adjust lifecycle dates, compare raw vs smooth inputs, correct labels, and save accepted crop examples for future data collection or retraining.
