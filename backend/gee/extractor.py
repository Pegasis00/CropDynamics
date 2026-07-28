"""
GEEDataExtractor — Sentinel-2 NDVI point extraction.

This is the live API version of the Sentinel-2 point NDVI extraction:
  * Collection: COPERNICUS/S2_SR_HARMONIZED
  * Exact point filtering and point sampling only.
  * No cloud masking, no QA60 masking — only scene-level filtering via
    CLOUDY_PIXEL_PERCENTAGE < cloud_threshold_pct.
  * Duplicate acquisition dates collapsed to the lowest-cloud image per date.
  * NDVI = normalizedDifference([nir_band, red_band]).
  * Date window: [query_date - months_before, query_date + months_after].

Only the NDVI value, acquisition date, and scene-level cloud percentage
are carried into the inference pipeline.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from backend.config.loader import GEEConfig
from backend.utils.exceptions import GEEExtractionError
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NDVIObservation:
    date: dt.date
    ndvi: float
    cloud_percentage: float


class GEEDataExtractor:
    """Wraps the Earth Engine Python API. `ee.Initialize()` must already
    have been called by the caller (e.g. at FastAPI startup) with valid
    service-account credentials."""

    def __init__(self, config: GEEConfig, ee_module=None):
        self.config = config
        # Lazily imported so the rest of the app can be unit-tested without
        # the `earthengine-api` package / credentials being present.
        if ee_module is None:
            import ee as ee_module  # noqa: WPS433 (intentional local import)
        self.ee = ee_module

    def extract(self, latitude: float, longitude: float, query_date: dt.date) -> list[NDVIObservation]:
        ee = self.ee
        cfg = self.config

        point = ee.Geometry.Point([longitude, latitude])
        gt_date = ee.Date(query_date.isoformat())
        start_date = gt_date.advance(-cfg.months_before, "month")
        end_date = gt_date.advance(cfg.months_after, "month")

        def tag_date(image):
            image = ee.Image(image)
            return image.set("Date", image.date().format("YYYY-MM-dd"))

        def add_ndvi(image):
            image = ee.Image(image)
            ndvi = image.normalizedDifference([cfg.ndvi_nir_band, cfg.ndvi_red_band]).rename("NDVI")
            return image.addBands(ndvi)

        collection = (
            ee.ImageCollection(cfg.collection)
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.cloud_threshold_pct))
            .map(tag_date)
        )

        # Sort by cloud first so distinct("Date") keeps the least-cloudy
        # image for each acquisition date without building one filtered
        # collection per date.
        collection = (
            collection.sort("CLOUDY_PIXEL_PERCENTAGE")
            .distinct("Date")
            .sort("system:time_start")
            .map(add_ndvi)
        )

        def sample_pixel(image):
            image = ee.Image(image)
            ndvi = image.select("NDVI").reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=cfg.pixel_scale_m,
                bestEffort=True,
                maxPixels=16,
            )
            return ee.Feature(
                None,
                {
                    "Image_Date": image.date().format("YYYY-MM-dd"),
                    "Cloud_Percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                    "NDVI": ndvi.get("NDVI"),
                },
            )

        samples = ee.FeatureCollection(collection.map(sample_pixel)).filter(ee.Filter.notNull(["NDVI"]))

        try:
            features = samples.getInfo()["features"]
            start_label = start_date.format("YYYY-MM-dd").getInfo()
            end_label = end_date.format("YYYY-MM-dd").getInfo()
        except Exception as exc:
            logger.exception(
                "Earth Engine NDVI extraction failed for (%.5f, %.5f) with %d/%d month window",
                latitude,
                longitude,
                cfg.months_before,
                cfg.months_after,
            )
            raise GEEExtractionError() from exc

        observations = [
            NDVIObservation(
                date=dt.datetime.strptime(f["properties"]["Image_Date"], "%Y-%m-%d").date(),
                ndvi=float(f["properties"]["NDVI"]),
                cloud_percentage=float(f["properties"]["Cloud_Percentage"]),
            )
            for f in features
        ]
        observations.sort(key=lambda o: o.date)

        logger.info(
            "Extracted %d NDVI observations for (%.5f, %.5f) window [%s, %s]",
            len(observations), latitude, longitude, start_label, end_label,
        )
        return observations
