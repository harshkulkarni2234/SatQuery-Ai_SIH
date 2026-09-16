# Scenario A — single image (VQA / grounding demo)

`single_image.jpg` — a satellite view of a city on a river, with visible
water (the river), built-up area (the city itself), and vegetation
(surrounding green land). Good for VQA presence questions ("Is there
water/vegetation/a built-up area in this image?") and grounding queries
("Where is the water located?").

**Source: unconfirmed.** This file was provided by the project owner in a
local `testing/` folder with no accompanying metadata, README, or license
file. It has no visible watermark, but I could not independently verify its
origin, exact acquisition date, or license from the file alone — treat any
specific claim about it (sensor, date, rights) as unverified until the
original source is found. Do not present it as a fully cited/licensed
asset in a public report without confirming its source first.

If you can identify the source (e.g. a specific Earth Observatory / Copernicus
/ USGS page), update this README with dataset name, license, citation and
source URL per the manifest convention in `data/manifest.json`.

## Alternative: a fully-cited image also exists

The same `testing/` folder also contains `farm2.jpg`, which carries visible
Copernicus/ESA attribution baked into the image: Sentinel-2 L2A True Color,
11 May 2023, Bełżyce, Poland (Copernicus Sentinel data, EU/ESA). It doesn't
show a large water body, but it's a safer choice if a fully-attributable
single image is required for anything public-facing (report, demo day).
