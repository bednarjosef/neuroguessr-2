# Benchmark test sets — provenance

Downloaded 2026-08-07. Two standard image-geolocation test benchmarks, each with
`images/` + `manifest.csv` (`path,lat,lon`, path relative to the dataset dir).
All images verified to open cleanly with PIL; all coords within valid ranges.

## im2gps/ — original Hays & Efros 2008 test set

- **Images**: 237 JPEGs from the official CMU project page:
  `http://graphics.cs.cmu.edu/projects/im2gps/gps_query_imgs.zip` (38.7 MB).
- **Ground truth**: `im2gps_places365.csv` (IMG_ID, LAT, LON; 237 rows), the standard
  TIBHannover/GeoEstimation meta file. The TIBHannover repo has been gutted (source and
  meta files removed for ethics reasons; even the `original_tf` branch tree is
  unreachable), so the CSV was taken from a research mirror:
  `https://raw.githubusercontent.com/ShramanPramanick/Transformer_Based_Geo-localization/main/resources/im2gps_places365.csv`
- **Join**: CMU filenames carry a `Placename_NNNNN_` prefix (e.g.
  `Rome_00024_231521295_...jpg`); GT IMG_ID is the bare Flickr suffix. Joined by unique
  suffix match — 237/237 matched, no losses.
- **Counts**: 237 images on disk = 237 manifest rows = canonical 237. Zero corrupt.

## im2gps3k/ — Vo et al. 2017 ("Revisiting IM2GPS") test set

- **Images**: `im2gps3ktest.zip` (477 MB, 3000 JPEGs, internal timestamps 2017-09-14 —
  the original Nam Vo release, canonically hosted at
  `http://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip`), mirrored on
  Hugging Face: dataset `MatchaMacchiato/img2gps3k`, file `im2gps3ktest.zip`.
- **Ground truth**: `im2gps3k_places365.csv` (2997 rows), the standard
  TIBHannover/GeoEstimation meta file, taken (via git-LFS) from mirror:
  `https://media.githubusercontent.com/media/Junchen-Ding/Geolocation/main/results/im2gps3k_places365.csv`
- **Counts**: 3000 images on disk; 2997 manifest rows = canonical eval count (the
  standard GT CSV covers 2997 of the 3000 zip images; the 3 without GT are excluded by
  the whole literature and are left on disk unmanifested:
  `199384251_7860f20c04_66_63163416@N00.jpg`,
  `253668751_c8cce71b58_102_23601949@N00.jpg`,
  `504935857_52562f0ee0_217_24311489@N00.jpg`). Zero corrupt, zero dead-link losses.

## Sanity checks (2026-08-07)

- All 3234 manifest images open + fully decode with PIL.
- All lat in [-90, 90], lon in [-180, 180].
- Landmark spot-checks: im2gps `Rome_00024_...` GT = 41.8904, 12.4920 (Colosseum, 0.0 km);
  nearest-to-landmark GT hits at ~0.0 km for Notre-Dame, Times Square, and Trafalgar
  Square in im2gps3k; Paris/Rome/London-tagged im2gps images all geolocate to their city.
- `_downloads/` keeps the two GT source CSVs; the image zips were deleted after
  extraction (re-downloadable from the URLs above).
