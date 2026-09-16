# Scenario B — temporal pair (change detection demo)

`before.tif` / `after.tif` — real Sentinel-2 L2A true-color crops of the same
location (Lake Mead / Hoover Dam area, Nevada–Arizona, USA), at two real,
different dates:

| | Scene ID | Date (UTC) | Cloud cover |
|---|---|---|---|
| before | `S2A_11SPV_20180824_1_L2A` | 2018-08-24 | 1.9% |
| after | `S2A_11SPV_20220823_0_L2A` | 2022-08-23 | 0.0% |

Both dates fall in the same calendar month (late August) in different years,
so seasonal/sun-angle differences are minimized — any real difference the
pipeline detects is more likely to reflect an actual change at the site
(this area is a well-documented drought/reservoir-drawdown location) than a
seasonal artifact. That said: **I have not independently verified the exact
magnitude of shoreline change visible in this specific crop.** A pixel-level
diff of the two files shows ~33% of pixels differ above a small threshold,
but the mean RGB values are nearly identical — meaning most of that is
texture/wave/shadow noise, not necessarily a large land-cover change within
this exact crop window. Don't caption this pair with a specific claimed
percentage of "water lost" unless SatQuery's own change-detection output
measures it — that's exactly the kind of claim the project's honesty rules
say never to fabricate.

## Source and reproducibility

Source: Sentinel-2 L2A Cloud-Optimized GeoTIFFs on the public AWS
"sentinel-cogs" bucket (`s3://sentinel-cogs`), discovered via the Element84
Earth Search STAC API (`https://earth-search.aws.element84.com/v1/search`).
No account or API key needed. Sentinel-2 data is Copernicus Sentinel data,
freely available under the
[Copernicus open data licence](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).

Reproduce with:

```bash
backend/venv_mac/bin/python ml/data/download_lakemead_temporal_pair.py
```

The script re-queries STAC for the lowest-cloud scene in each date window
and crops only the small AOI via a windowed COG read (a few MB transferred,
not the full ~100MB+ Sentinel-2 tile) — no manual download step needed.

## Known limitation

Both GeoTIFFs are Sentinel-2 **true-color (TCI) visualization products**,
not raw reflectance bands — they carry no embedded acquisition-date TIFF tag
(SatQuery's `extract_metadata()` correctly reports
`acquisition_date_source: "unknown"` for them as delivered). The real dates
are known from the STAC query above (2018-08-24 / 2022-08-23) — when
uploading these through the app for a demo, supply them explicitly via the
upload form's capture-date field so they're recorded honestly as
`acquisition_date_source: "user"` rather than left unknown.
