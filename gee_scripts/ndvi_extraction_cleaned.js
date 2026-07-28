//====================================================
// SECTION 1 : USER SETTINGS
//====================================================
var MONTHS = ee.Dictionary({
  'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
  'Jul': '07', 'Aug': '08', 'Sep': '09', 'Sept':'09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
  'January':'01', 'February':'02', 'March':'03', 'April':'04', 'June':'06',
  'July':'07', 'August':'08', 'September':'09', 'October':'10', 'November':'11', 'December':'12'
});

var farms = Farms; // input FeatureCollection
// required columns: GroundTruth_Date, Latitude, Longitude

var CLOUD_LIMIT   = 20; // max scene-level cloud %
var PIXEL_SCALE     = 10; // Sentinel-2 native resolution
var MONTHS_BEFORE    = 6; // window before ground-truth date
var MONTHS_AFTER      = 6; // window after ground-truth date

//====================================================
// SECTION 2 : DATE HELPERS
//====================================================

// Parse "Jul-22" -> 2022-07-01
function getGroundTruthDate(farm){
  var parts = ee.String(farm.get('GroundTruth_Date')).split('-');
  var month = ee.String(parts.get(0));
  var yearShort = ee.String(parts.get(1)); // e.g. "22"
  var year = ee.String('20').cat(yearShort); // -> "2022"
  var monthNum = ee.String(MONTHS.get(month));

  return ee.Date.parse(
    'yyyy-MM-dd',
    year.cat('-').cat(monthNum).cat('-01')
  );
}

//====================================================
// SECTION 3 : IMAGE HELPERS
//====================================================

// Tag every image with its acquisition date string (needed for dedup)
function tagDate(image) {
  image = ee.Image(image);
  return image.set('Date', image.date().format('YYYY-MM-dd'));
}

// NDVI = (B8 - B4) / (B8 + B4), original bands kept
function addNDVI(image) {
  image = ee.Image(image);
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}

//====================================================
// SECTION 4 : PER-FARM PROCESSING
//====================================================
function processFarm(farm) {
  farm = ee.Feature(farm);

  // Point is built directly from the Latitude/Longitude columns —
  // this is the only geometry used anywhere in the script.
  var lat = ee.Number.parse(farm.get('Latitude'));
  var lon = ee.Number.parse(farm.get('Longitude'));
  var point = ee.Geometry.Point([lon, lat]);

  var gtDate = getGroundTruthDate(farm);
  var startDate = gtDate.advance(-MONTHS_BEFORE, 'month');
  var endDate = gtDate.advance(MONTHS_AFTER, 'month');

  //--------------------------------------------------
  // Sentinel-2 collection, filtered to the farm and date range
  //--------------------------------------------------
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(point)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT))
    .map(tagDate);

  //--------------------------------------------------
  // Duplicate acquisition dates: keep the lowest-cloud image per date
  //--------------------------------------------------
  collection = collection
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .distinct('Date')
    .sort('system:time_start');

  //--------------------------------------------------
  // NDVI (no masking applied — manual QA happens after export)
  //--------------------------------------------------
  collection = collection.map(addNDVI);

  //--------------------------------------------------
  // Extract the NDVI value at the exact pixel under point.
  //--------------------------------------------------
  var samples = ee.FeatureCollection(collection.map(function(image) {
    image = ee.Image(image);
    var ndvi = image.select('NDVI').reduceRegion({
      reducer: ee.Reducer.first(),
      geometry: point,
      scale: PIXEL_SCALE,
      bestEffort: true,
      maxPixels: 16
    });

    return ee.Feature(null, {
      Farm_ID: farm.id(),
      NDVI: ndvi.get('NDVI'),
      GroundTruth_Date: farm.get('GroundTruth_Date'),
      Latitude: lat,
      Longitude: lon,
      Image_Date: image.date().format('YYYY-MM-dd'),
      Cloud_Percentage: image.get('CLOUDY_PIXEL_PERCENTAGE'),
      Extraction_Start: startDate.format('YYYY-MM-dd'),
      Extraction_End: endDate.format('YYYY-MM-dd')
    });
  })).filter(ee.Filter.notNull(['NDVI']));

  return samples;
}

//====================================================
// SECTION 5 : RUN OVER ALL FARMS
//====================================================
var farmList = farms.toList(farms.size());
var allSamples = ee.FeatureCollection(farmList.map(processFarm)).flatten();

//====================================================
// SECTION 6 : EXPORT
//====================================================
Export.table.toDrive({
  collection: allSamples,
  description: 'NDVI_TimeSeries',
  fileNamePrefix: 'NDVI_TimeSeries',
  folder: 'GEE',
  fileFormat: 'CSV',
  selectors: [
    'Farm_ID',
    'GroundTruth_Date',
    'Latitude',
    'Longitude',
    'Extraction_Start',
    'Extraction_End',
    'Image_Date',
    'NDVI',
    'Cloud_Percentage'
  ]
});
