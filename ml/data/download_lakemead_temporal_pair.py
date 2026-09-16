"""Download a genuine before/after Sentinel-2 pair over Lake Mead, NV/AZ.

Reproduces data/demo/scenario_B_temporal/{before,after}.tif: two real Sentinel-2
L2A true-color (TCI) crops of the same location at different real dates,
fetched from the public AWS "sentinel-cogs" bucket (Element84 Earth Search
STAC API for scene discovery, then a windowed read of the Cloud-Optimized
GeoTIFF so only the small AOI is actually transferred — a few MB total, not
the ~100MB+ full scene).

No account or API key required. Both dates are picked automatically as the
lowest-cloud-cover Sentinel-2 scene within each search window; if AWS/STAC
availability changes, re-run this script — it will pick whatever the
lowest-cloud match is at run time, so exact scene IDs may differ from the
ones documented in data/manifest.json for the committed copies.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

import requests

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
except ImportError:
    print("This script needs rasterio. Run it with the backend venv:", file=sys.stderr)
    print("  backend/venv_mac/bin/python ml/data/download_lakemead_temporal_pair.py", file=sys.stderr)
    raise

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

# Small AOI around the Hoover Dam / Lake Mead marina area (WGS84 lon/lat).
AOI_WGS84 = (-114.78, 36.00, -114.60, 36.15)

# Same calendar month in two different years so sun angle/season match,
# letting any real difference be about water level rather than season.
SEARCH_WINDOWS = {
    "before": ("2018-06-01T00:00:00Z", "2018-08-31T23:59:59Z"),
    "after": ("2022-06-01T00:00:00Z", "2022-08-31T23:59:59Z"),
}

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "demo", "scenario_B_temporal"
)


def _find_lowest_cloud_scene(start: str, end: str) -> dict:
    resp = requests.post(
        STAC_URL,
        json={
            "collections": [COLLECTION],
            "bbox": list(AOI_WGS84),
            "datetime": f"{start}/{end}",
            "query": {"eo:cloud_cover": {"lt": 10}},
            "limit": 20,
        },
        timeout=30,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise RuntimeError(f"No low-cloud Sentinel-2 scene found for {start}..{end}")
    return min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))


def _crop_visual_asset(feature: dict, out_path: str) -> dict:
    url = feature["assets"]["visual"]["href"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(url) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *AOI_WGS84)
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=src.window_transform(window),
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
    return {
        "scene_id": feature["id"],
        "datetime": feature["properties"]["datetime"],
        "cloud_cover": feature["properties"].get("eo:cloud_cover"),
        "source_asset": url,
    }


def main() -> None:
    manifest_entries = {}
    for name, (start, end) in SEARCH_WINDOWS.items():
        print(f"Searching for lowest-cloud Sentinel-2 scene: {name} ({start[:10]}..{end[:10]})")
        feature = _find_lowest_cloud_scene(start, end)
        out_path = os.path.join(OUT_DIR, f"{name}.tif")
        info = _crop_visual_asset(feature, out_path)
        manifest_entries[name] = info
        print(f"  -> {info['scene_id']} ({info['datetime']}, cloud {info['cloud_cover']:.1f}%) -> {out_path}")

    print("\nDone. Verify with:")
    print("  backend/venv_mac/bin/python -c \"import rasterio; "
          "print(rasterio.open('data/demo/scenario_B_temporal/before.tif').crs)\"")
    print(json.dumps(manifest_entries, indent=2))


if __name__ == "__main__":
    main()
