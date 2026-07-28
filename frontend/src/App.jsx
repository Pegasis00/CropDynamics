import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Chart from "chart.js/auto";
import { getRelativePosition } from "chart.js/helpers";

const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://localhost:8000/api"
    : "/api");

const chartTitles = {
  raw: "Raw and smoothed NDVI",
  markers: "Detected peaks and valleys",
  lifecycle: "Selected lifecycle",
};

const fallbackCropOptions = ["Cotton", "Onion", "Paddy", "Sugarcane", "Wheat"];

function errorMessageFromBody(body) {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail) && body.detail.length > 0) {
    return body.detail.map((err) => err.msg || String(err)).join(" ");
  }
  return "Something went wrong.";
}

function parseCoordinatePair(value) {
  const parts = String(value || "").match(/[-+]?\d+(?:\.\d+)?/g) || [];
  if (parts.length < 2) return null;
  return {
    latitude: Number.parseFloat(parts[0]),
    longitude: Number.parseFloat(parts[1]),
  };
}

function chartLabels(plot) {
  const labels = [...(plot?.dates || [])];
  if (plot?.query_date && !labels.includes(plot.query_date)) {
    labels.push(plot.query_date);
    labels.sort();
  }
  return labels;
}

function seriesPoints(dates, values) {
  return (dates || []).map((date, index) => ({ x: date, y: values[index] }));
}

function queryDateLineDataset(plot) {
  if (!plot?.query_date) return null;
  return {
    label: "Query date",
    data: [
      { x: plot.query_date, y: 0 },
      { x: plot.query_date, y: 1 },
    ],
    borderColor: "#c28d2c",
    borderWidth: 2,
    pointRadius: 0,
    borderDash: [6, 4],
    tension: 0,
  };
}

function pickDateFromChartClick(chart, event, labels) {
  const canvasPosition = getRelativePosition(event, chart);
  const rawIndex = chart.scales.x.getValueForPixel(canvasPosition.x);
  const index = Math.max(0, Math.min(labels.length - 1, Math.round(Number(rawIndex))));
  return labels[index] || "";
}

function buildDatasets(kind, plot) {
  const baseRaw = {
    label: "Raw NDVI",
    data: seriesPoints(plot.dates, plot.raw_ndvi),
    borderColor: "#8fa39a",
    backgroundColor: "rgba(143, 163, 154, 0.12)",
    borderDash: [4, 3],
    pointRadius: 2,
    tension: 0.18,
  };
  const baseSmooth = {
    label: "Smoothed NDVI",
    data: seriesPoints(plot.dates, plot.smoothed_ndvi),
    borderColor: "#24745c",
    backgroundColor: "rgba(36,116,92,0.12)",
    borderWidth: 2.5,
    pointRadius: 0,
    tension: 0.25,
  };

  const datasets = kind === "markers" ? [baseSmooth] : [baseRaw, baseSmooth];

  if (kind === "markers") {
    datasets.push({
      label: "Peak",
      type: "scatter",
      data: (plot.peaks || []).map((point) => ({ x: point.date, y: point.ndvi })),
      backgroundColor: "#a84d3f",
      borderColor: "#a84d3f",
      pointStyle: "triangle",
      pointRadius: 7,
    });
    datasets.push({
      label: "Valley",
      type: "scatter",
      data: (plot.valleys || []).map((point) => ({ x: point.date, y: point.ndvi })),
      backgroundColor: "#c28d2c",
      borderColor: "#c28d2c",
      pointStyle: "rectRot",
      pointRadius: 6,
    });
  }

  if (kind === "lifecycle" && plot.lifecycle) {
    const selectedFeatureSet = plot.lifecycle.feature_set || "smooth";
    const selectedNdvi = plot.lifecycle.span_selected_ndvi || plot.lifecycle.span_smooth_ndvi;
    datasets.push({
      label: `Selected ${selectedFeatureSet} lifecycle`,
      data: plot.lifecycle.span_dates.map((date, index) => ({
        x: date,
        y: selectedNdvi[index],
      })),
      borderColor: selectedFeatureSet === "raw" ? "#a84d3f" : "#355f8c",
      backgroundColor: selectedFeatureSet === "raw" ? "rgba(168,77,63,0.12)" : "rgba(53,95,140,0.14)",
      borderWidth: 4,
      pointRadius: 0,
      fill: true,
      tension: 0.2,
    });
    datasets.push({
      label: "SOS / Peak / EOS",
      type: "scatter",
      data: [
        { x: plot.lifecycle.sos.date, y: plot.lifecycle.sos.ndvi },
        { x: plot.lifecycle.peak.date, y: plot.lifecycle.peak.ndvi },
        { x: plot.lifecycle.eos.date, y: plot.lifecycle.eos.ndvi },
      ],
      backgroundColor: "#17211c",
      borderColor: "#17211c",
      pointRadius: 7,
      pointStyle: ["rect", "triangle", "rect"],
    });
  }

  const queryLine = queryDateLineDataset(plot);
  if (queryLine) datasets.push(queryLine);
  return datasets;
}

function SignalChart({ activeChart, data, editingTarget, onDatePick }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  const plot = useMemo(() => {
    if (!data) return null;
    const key =
      activeChart === "raw"
        ? "raw_vs_smoothed"
        : activeChart === "markers"
          ? "signal_with_markers"
          : "selected_lifecycle";
    return data.plots[key];
  }, [activeChart, data]);

  useEffect(() => {
    if (!plot || !canvasRef.current) return undefined;

    if (chartRef.current) chartRef.current.destroy();

    const labels = chartLabels(plot);

    chartRef.current = new Chart(canvasRef.current, {
      type: "line",
      data: {
        labels,
        datasets: buildDatasets(activeChart, plot),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { labels: { usePointStyle: true, boxWidth: 8, font: { weight: "600" } } },
          tooltip: { backgroundColor: "#17211c", padding: 12 },
        },
        scales: {
          x: {
            type: "category",
            grid: { display: false },
            ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 9 },
          },
          y: {
            min: 0,
            max: 1,
            grid: { color: "rgba(99,112,104,0.18)" },
            ticks: { stepSize: 0.2 },
          },
        },
        onClick: (event) => {
          if (!editingTarget || activeChart !== "lifecycle") return;
          const pickedDate = pickDateFromChartClick(chartRef.current, event, labels);
          if (pickedDate) onDatePick(editingTarget, pickedDate);
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [activeChart, editingTarget, onDatePick, plot]);

  return <canvas ref={canvasRef} />;
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function chartTitle(activeChart, featureSet) {
  if (activeChart !== "lifecycle") return chartTitles[activeChart];
  return featureSet === "raw" ? "Selected raw lifecycle" : "Selected smooth lifecycle";
}

export default function App() {
  const [coords, setCoords] = useState("");
  const [lat, setLat] = useState("18.5204");
  const [lon, setLon] = useState("73.8567");
  const [queryDate, setQueryDate] = useState("2024-10-15");
  const [status, setStatus] = useState({ kind: "ready", text: "Ready" });
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [featureSet, setFeatureSet] = useState("smooth");
  const [activeChart, setActiveChart] = useState("raw");
  const [loading, setLoading] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [editTarget, setEditTarget] = useState("sos");
  const [editValues, setEditValues] = useState({ query_date: "", sos_date: "", eos_date: "" });
  const [manualLoading, setManualLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);
  const [resultLocation, setResultLocation] = useState(null);
  const [cropOptions, setCropOptions] = useState(fallbackCropOptions);
  const [labelEditMode, setLabelEditMode] = useState(false);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [reviewedCropLabel, setReviewedCropLabel] = useState("");

  useEffect(() => {
    let ignore = false;
    fetch(`${API_BASE}/crops`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (!ignore && Array.isArray(payload?.crops) && payload.crops.length > 0) {
          setCropOptions(payload.crops);
        }
      })
      .catch(() => {});
    return () => {
      ignore = true;
    };
  }, []);

  async function runPrediction(event) {
    event.preventDefault();
    setError("");

    const parsedLat = Number.parseFloat(lat);
    const parsedLon = Number.parseFloat(lon);
    if ((!coords.trim() && (Number.isNaN(parsedLat) || Number.isNaN(parsedLon))) || !queryDate) {
      setError("Please fill in coordinates or latitude/longitude, plus a query date.");
      setStatus({ kind: "error", text: "Error" });
      return;
    }

    const parsedCoords = coords.trim() ? parseCoordinatePair(coords) : null;
    if (coords.trim() && (!parsedCoords || Number.isNaN(parsedCoords.latitude) || Number.isNaN(parsedCoords.longitude))) {
      setError("Please paste coordinates as latitude, longitude.");
      setStatus({ kind: "error", text: "Error" });
      return;
    }

    const requestLocation = coords.trim()
      ? parsedCoords
      : { latitude: parsedLat, longitude: parsedLon };

    setLoading(true);
    setStatus({ kind: "busy", text: "Extracting NDVI" });

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          coords.trim()
            ? { coordinates: coords.trim(), query_date: queryDate, feature_set: featureSet }
            : { latitude: parsedLat, longitude: parsedLon, query_date: queryDate, feature_set: featureSet }
        ),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Something went wrong." }));
        setError(errorMessageFromBody(body));
        setStatus({ kind: "error", text: "Error" });
        return;
      }

      const payload = await response.json();
      setResult(payload);
      setFeatureSet(payload.feature_set || featureSet);
      setResultLocation({ ...requestLocation, query_date: queryDate });
      setActiveChart("raw");
      setEditMode(false);
      setEditTarget("sos");
      setEditValues({ query_date: "", sos_date: "", eos_date: "" });
      setSaveLoading(false);
      setLabelEditMode(false);
      setSelectedLabel(payload.predicted_crop);
      setReviewedCropLabel("");
      setStatus({ kind: "ready", text: "Prediction complete" });
    } catch {
      setError("Could not reach the prediction service. Is the backend running?");
      setStatus({ kind: "error", text: "Error" });
    } finally {
      setLoading(false);
    }
  }

  function clearResult() {
    setResult(null);
    setError("");
    setResultLocation(null);
    setSaveLoading(false);
    setLabelEditMode(false);
    setSelectedLabel("");
    setReviewedCropLabel("");
    setEditMode(false);
    setEditTarget("sos");
    setEditValues({ query_date: "", sos_date: "", eos_date: "" });
    setStatus({ kind: "ready", text: "Ready" });
  }

  function startLifecycleEdit() {
    if (!result?.lifecycle) return;
    setEditValues({
      query_date: result.plots?.selected_lifecycle?.query_date || queryDate,
      sos_date: result.lifecycle.sos_date,
      eos_date: result.lifecycle.eos_date,
    });
    setEditTarget("sos");
    setActiveChart("lifecycle");
    setEditMode(true);
    setError("");
  }

  const handleLifecycleDatePick = useCallback((target, date) => {
    setEditValues((current) => ({
      ...current,
      [`${target}_date`]: date,
    }));
  }, []);

  async function runManualLifecyclePrediction() {
    const plot = result?.plots?.selected_lifecycle;
    if (!plot || !editValues.query_date || !editValues.sos_date || !editValues.eos_date) {
      setError("Choose a query date, SOS, and EOS before rerunning the prediction.");
      return;
    }
    if (editValues.eos_date <= editValues.sos_date) {
      setError("EOS must be after SOS.");
      return;
    }

    setManualLoading(true);
    setError("");
    setStatus({ kind: "busy", text: "Rerunning edited lifecycle" });

    try {
      const response = await fetch(`${API_BASE}/predict/manual-lifecycle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dates: plot.dates,
          raw_ndvi: plot.raw_ndvi,
          smoothed_ndvi: plot.smoothed_ndvi,
          query_date: editValues.query_date,
          sos_date: editValues.sos_date,
          eos_date: editValues.eos_date,
          feature_set: featureSet,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Something went wrong." }));
        setError(errorMessageFromBody(body));
        setStatus({ kind: "error", text: "Error" });
        return;
      }

      const payload = await response.json();
      setResult(payload);
      setFeatureSet(payload.feature_set || featureSet);
      setActiveChart("lifecycle");
      setSaveLoading(false);
      setLabelEditMode(false);
      setSelectedLabel(payload.predicted_crop);
      setReviewedCropLabel("");
      setEditValues({
        query_date: payload.plots?.selected_lifecycle?.query_date || editValues.query_date,
        sos_date: payload.lifecycle?.sos_date || editValues.sos_date,
        eos_date: payload.lifecycle?.eos_date || editValues.eos_date,
      });
      setStatus({ kind: "ready", text: "Edited prediction complete" });
    } catch {
      setError("Could not reach the prediction service. Is the backend running?");
      setStatus({ kind: "error", text: "Error" });
    } finally {
      setManualLoading(false);
    }
  }

  async function runFeatureSetPrediction(nextFeatureSet) {
    const plot = result?.plots?.selected_lifecycle;
    if (!result?.lifecycle || !plot) {
      setFeatureSet(nextFeatureSet);
      return;
    }

    setManualLoading(true);
    setError("");
    setStatus({ kind: "busy", text: nextFeatureSet === "raw" ? "Testing raw data" : "Testing smooth data" });

    try {
      const response = await fetch(`${API_BASE}/predict/manual-lifecycle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dates: plot.dates,
          raw_ndvi: plot.raw_ndvi,
          smoothed_ndvi: plot.smoothed_ndvi,
          query_date: plot.query_date || resultLocation?.query_date || queryDate,
          sos_date: result.lifecycle.sos_date,
          eos_date: result.lifecycle.eos_date,
          feature_set: nextFeatureSet,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Something went wrong." }));
        setError(errorMessageFromBody(body));
        setStatus({ kind: "error", text: "Error" });
        return;
      }

      const payload = await response.json();
      setResult(payload);
      setFeatureSet(payload.feature_set || nextFeatureSet);
      setActiveChart("lifecycle");
      setSaveLoading(false);
      setLabelEditMode(false);
      setSelectedLabel(payload.predicted_crop);
      setReviewedCropLabel("");
      setEditValues({
        query_date: payload.plots?.selected_lifecycle?.query_date || plot.query_date || queryDate,
        sos_date: payload.lifecycle?.sos_date || result.lifecycle.sos_date,
        eos_date: payload.lifecycle?.eos_date || result.lifecycle.eos_date,
      });
      setStatus({ kind: "ready", text: `${nextFeatureSet === "raw" ? "Raw" : "Smooth"} prediction complete` });
    } catch {
      setError("Could not reach the prediction service. Is the backend running?");
      setStatus({ kind: "error", text: "Error" });
    } finally {
      setManualLoading(false);
    }
  }

  async function savePrediction() {
    const lifecycle = result?.plots?.selected_lifecycle?.lifecycle;
    const cropLabelForSave = reviewedCropLabel || result?.predicted_crop || "";
    if (!result || !lifecycle || !cropOptions.includes(cropLabelForSave)) {
      setError("Choose a crop label and make sure a lifecycle is available before saving.");
      return;
    }

    setSaveLoading(true);
    setError("");
    setStatus({ kind: "busy", text: "Saving prediction" });

    try {
      const response = await fetch(`${API_BASE}/predictions/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop_label: cropLabelForSave,
          dates: lifecycle.span_dates,
          raw_ndvi: lifecycle.span_raw_ndvi,
          query_date: result.plots?.selected_lifecycle?.query_date || resultLocation?.query_date || queryDate,
          latitude: resultLocation?.latitude ?? null,
          longitude: resultLocation?.longitude ?? null,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ detail: "Something went wrong." }));
        setError(errorMessageFromBody(body));
        setStatus({ kind: "error", text: "Error" });
        return;
      }

      const payload = await response.json();
      setResult((current) => (current ? { ...current, saved_crop_id: payload.saved_crop_id } : current));
      setStatus({ kind: "ready", text: "Prediction saved" });
    } catch {
      setError("Could not reach the prediction service. Is the backend running?");
      setStatus({ kind: "error", text: "Error" });
    } finally {
      setSaveLoading(false);
    }
  }

  function startLabelEdit() {
    if (!result) return;
    setSelectedLabel(reviewedCropLabel || (cropOptions.includes(result.predicted_crop) ? result.predicted_crop : cropOptions[0] || ""));
    setLabelEditMode(true);
    setError("");
  }

  function confirmLabelEdit() {
    if (!selectedLabel) {
      setError("Choose a crop label before confirming.");
      return;
    }

    setReviewedCropLabel(selectedLabel);
    setResult((current) => (current ? { ...current, saved_crop_id: null } : current));
    setLabelEditMode(false);
    setStatus({ kind: "ready", text: "Prediction label updated" });
  }

  const displayedCropLabel = reviewedCropLabel || result?.predicted_crop || "";
  const hasReviewerLabel = Boolean(reviewedCropLabel && reviewedCropLabel !== result?.predicted_crop);
  const canSavePrediction = Boolean(result?.lifecycle && cropOptions.includes(displayedCropLabel));

  const note = useMemo(() => {
    if (!result) return "";
    if (result.predicted_crop === "No plant found") {
      return "The NDVI lifecycle is too weak or flat to treat as a crop signal.";
    }
    if (!result.lifecycle) {
      return "The NDVI series is available, but no valid crop lifecycle was extracted for this location and date.";
    }
    if (hasReviewerLabel) {
      return result.saved_crop_id
        ? `Saved corrected ${reviewedCropLabel} lifecycle as ${result.saved_crop_id}.`
        : `Corrected label set to ${reviewedCropLabel}. Prediction is not saved yet.`;
    }
    if (result.is_other_crop) {
      return result.lifecycle
        ? "The lifecycle was extracted, but confidence was below the trained crop threshold."
        : "The NDVI series is available, but no valid crop lifecycle was extracted for this location and date.";
    }
    return result.saved_crop_id
      ? `Saved SOS-to-EOS NDVI points as ${result.saved_crop_id}.`
      : "Prediction is not saved yet.";
  }, [hasReviewerLabel, result, reviewedCropLabel]);

  return (
    <main className="shell">
      <aside className="rail">
        <div className="brand">
          <div className="mark">FS</div>
          <div>
            <h1>Field Signal</h1>
            <span>Point NDVI crop inference</span>
          </div>
        </div>

        <form className="form-block" onSubmit={runPrediction}>
          <div className="field">
            <label htmlFor="coords">Coordinates</label>
            <input
              id="coords"
              type="text"
              value={coords}
              placeholder="18.5204, 73.8567"
              autoComplete="off"
              onChange={(event) => setCoords(event.target.value)}
            />
          </div>

          <div className="split">
            <div className="field">
              <label htmlFor="lat">Latitude</label>
              <input id="lat" type="number" step="0.0001" value={lat} onChange={(event) => setLat(event.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="lon">Longitude</label>
              <input id="lon" type="number" step="0.0001" value={lon} onChange={(event) => setLon(event.target.value)} />
            </div>
          </div>

          <div className="field">
            <label htmlFor="qdate">Query date</label>
            <input id="qdate" type="date" value={queryDate} onChange={(event) => setQueryDate(event.target.value)} />
          </div>

          <div className="field">
            <label>Model input</label>
            <div className="mode-toggle">
              <button
                className={`mode-button ${featureSet === "smooth" ? "active" : ""}`}
                type="button"
                onClick={() => setFeatureSet("smooth")}
              >
                Smooth
              </button>
              <button
                className={`mode-button ${featureSet === "raw" ? "active" : ""}`}
                type="button"
                onClick={() => setFeatureSet("raw")}
              >
                Test raw data
              </button>
            </div>
          </div>

          <button className="primary" type="submit" disabled={loading}>
            {loading ? "Running..." : "Run prediction"}
          </button>
          <button className="secondary" type="button" onClick={clearResult}>
            Clear result
          </button>
        </form>

        <p className="hint">
          Paste a coordinate pair or use separate latitude and longitude. The backend extracts exact-point Sentinel-2 NDVI and
          returns lifecycle plots.
        </p>

        {error && <div className="notice error">{error}</div>}

        <div className="rail-foot">
          <div className="status-line">
            <span className={`dot ${status.kind}`} />
            <span>{status.text}</span>
          </div>
          <p className="hint">Accepted crop predictions can be saved into per-crop CSV files after review.</p>
        </div>
      </aside>

      <section className="workspace">
        <div className="topbar">
          <div>
            <p className="eyebrow">Prediction workspace</p>
            <h2>NDVI signal, lifecycle, and crop decision in one view.</h2>
          </div>
        </div>

        {!result ? (
          <div className="empty">
            <div>
              <strong>No query loaded</strong>
              Run a prediction to see the extracted NDVI series, lifecycle markers, model result, and saved crop ID.
            </div>
          </div>
        ) : (
          <div className="result-stack">
            <div className="summary">
              <section className="result-pane">
                <div>
                  <p className="eyebrow">Model result</p>
                  <h3 className="prediction-label">{displayedCropLabel}</h3>
                  {hasReviewerLabel && <p className="review-note">Model predicted {result.predicted_crop}</p>}
                </div>
                <span className={`confidence ${result.confidence < 0.6 ? "low" : ""}`}>
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
                <span className="mode-badge">{featureSet === "raw" ? "Raw data model" : "Smooth data model"}</span>
                {result.lifecycle && (
                  <div className="review-actions">
                    {labelEditMode ? (
                      <div className="label-editor">
                        <div className="field">
                          <label htmlFor="crop-label">Crop label</label>
                          <select
                            id="crop-label"
                            value={selectedLabel}
                            onChange={(event) => setSelectedLabel(event.target.value)}
                          >
                            {cropOptions.map((crop) => (
                              <option key={crop} value={crop}>
                                {crop}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="label-buttons">
                          <button className="primary compact" type="button" onClick={confirmLabelEdit}>
                            Confirm
                          </button>
                          <button className="secondary compact" type="button" onClick={() => setLabelEditMode(false)}>
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button className="secondary compact save-action" type="button" onClick={startLabelEdit}>
                        Change prediction
                      </button>
                    )}
                    {canSavePrediction && (
                      <button
                        className="primary compact save-action"
                        type="button"
                        onClick={savePrediction}
                        disabled={saveLoading || Boolean(result.saved_crop_id)}
                      >
                        {result.saved_crop_id ? "Saved" : saveLoading ? "Saving..." : "Save to CSV"}
                      </button>
                    )}
                  </div>
                )}
              </section>

              <section className="metrics-pane">
                <Metric label="SOS" value={result.lifecycle?.sos_date} />
                <Metric label="Peak" value={result.lifecycle?.peak_date} />
                <Metric label="EOS" value={result.lifecycle?.eos_date} />
                <Metric label="Duration" value={result.lifecycle ? `${result.lifecycle.duration_days} days` : "-"} />
                <Metric label="Saved ID" value={result.saved_crop_id} />
              </section>
            </div>

            {note && <div className="notice active">{note}</div>}

            <section className="chart-pane">
              <div className="chart-head">
                <div>
                  <p className="eyebrow">Signal view</p>
                  <h3>{chartTitle(activeChart, featureSet)}</h3>
                </div>
                <div className="chart-actions">
                  {result.lifecycle && (
                    <div className="mode-toggle inline-mode">
                      <button
                        className={`mode-button ${featureSet === "smooth" ? "active" : ""}`}
                        type="button"
                        onClick={() => runFeatureSetPrediction("smooth")}
                        disabled={manualLoading || featureSet === "smooth"}
                      >
                        Smooth
                      </button>
                      <button
                        className={`mode-button ${featureSet === "raw" ? "active" : ""}`}
                        type="button"
                        onClick={() => runFeatureSetPrediction("raw")}
                        disabled={manualLoading || featureSet === "raw"}
                      >
                        Raw
                      </button>
                    </div>
                  )}
                  {result.lifecycle && (
                    <button
                      className={`secondary compact ${editMode ? "active" : ""}`}
                      type="button"
                      onClick={editMode ? () => setEditMode(false) : startLifecycleEdit}
                    >
                      {editMode ? "Done editing" : "Edit lifecycle"}
                    </button>
                  )}
                  <div className="tabs" role="tablist" aria-label="Chart views">
                    {Object.keys(chartTitles).map((chartKey) => (
                      <button
                        className={`tab ${activeChart === chartKey ? "active" : ""}`}
                        key={chartKey}
                        type="button"
                        onClick={() => setActiveChart(chartKey)}
                      >
                        {chartKey === "raw" ? "Raw" : chartKey === "markers" ? "Markers" : "Lifecycle"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              {editMode && (
                <div className="edit-panel">
                  <div className="field">
                    <label htmlFor="manual-qdate">Query date</label>
                    <input
                      id="manual-qdate"
                      type="date"
                      value={editValues.query_date}
                      onChange={(event) => setEditValues((current) => ({ ...current, query_date: event.target.value }))}
                    />
                  </div>
                  <div className="edit-controls">
                    <button
                      className={`choice ${editTarget === "sos" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        setEditTarget("sos");
                        setActiveChart("lifecycle");
                      }}
                    >
                      SOS {editValues.sos_date || "-"}
                    </button>
                    <button
                      className={`choice ${editTarget === "eos" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        setEditTarget("eos");
                        setActiveChart("lifecycle");
                      }}
                    >
                      EOS {editValues.eos_date || "-"}
                    </button>
                  </div>
                  <button className="primary compact" type="button" onClick={runManualLifecyclePrediction} disabled={manualLoading}>
                    {manualLoading ? "Running..." : "Run prediction"}
                  </button>
                </div>
              )}
              <div className="chart-body">
                <SignalChart
                  activeChart={activeChart}
                  data={result}
                  editingTarget={editMode ? editTarget : ""}
                  onDatePick={handleLifecycleDatePick}
                />
              </div>
            </section>
          </div>
        )}
      </section>
    </main>
  );
}
