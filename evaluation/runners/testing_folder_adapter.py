"""Dataset adapter over the local testing/ folder (see Phase B1's inventory
in docs/SOLO_PROGRESS.md and data/manifest.json's "unused_raw_material"
entry): ~4,000 BigEarthNet-style Sentinel-2 optical patches (2 tiles/dates,
no labels), 1 genuine Sentinel-1 SAR image, and a handful of misc single
images.

IMPORTANT — this is NOT RSVQA/CDVQA/VRSBench. Those benchmarks (Phases
B3-B6 as originally planned) ship labeled question/answer/box ground
truth; nothing in testing/ has any label file at all (verified: zero
.json/.csv/.txt/label files anywhere under testing/). The user explicitly
redirected this evaluation work to use testing/ instead of downloading
those external datasets. Consequently this adapter can only support an
UNLABELED capability/smoke-test run — it records real system behavior
(task routing, specialist selection, latency, confidence availability,
error rate) against real images, but it cannot and does not compute
accuracy against ground truth, because none exists in this data. See
evaluation/README.md for why this is not a scored benchmark.
"""

from __future__ import annotations

import os
import random
from typing import Iterator

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TESTING_DIR = os.path.join(REPO_ROOT, "testing")

# Excluded per Phase B1's documented findings (data/manifest.json) — kept
# consistent with the demo-data exclusions rather than re-litigated here.
EXCLUDED_FILENAMES = {
    "farm1.jpg",  # visible DigitalGlobe watermark — licensing risk
    "vqa1.jpg",   # uncropped Google Earth Engine Code Editor screenshot, not satellite imagery
}

SAR_FILENAME_PREFIX = "S1"  # e.g. S1A_IW_GRDH_... — Sentinel-1 SAR product naming

# Fixed, honestly-unscored prompts — same two used by the demo scenario A
# cards (Phase C9), applied here at scale as a capability smoke test, not
# a labeled benchmark question set.
PROMPTS = [
    "What can you tell me about this image?",
    "Show me the vegetation.",
]


def list_candidate_files() -> list[str]:
    if not os.path.isdir(TESTING_DIR):
        return []
    files = []
    for name in sorted(os.listdir(TESTING_DIR)):
        if name in EXCLUDED_FILENAMES:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        files.append(os.path.join(TESTING_DIR, name))
    return files


def _modality_for(path: str) -> str:
    name = os.path.basename(path)
    return "SAR" if name.startswith(SAR_FILENAME_PREFIX) else "OPTICAL"


def stratified_sample(files: list[str], n: int, seed: int = 42) -> list[str]:
    """Deterministic, seeded random sample — documented and reproducible,
    per the plan's own guidance for when a full run is too slow/large."""
    if n >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, n)


def generate_samples(n_images: int = 100, seed: int = 42) -> Iterator[dict]:
    """Yields one sample per (image, prompt) pair from a seeded random
    subset of testing/'s real images. `meta` carries provenance only —
    there is no ground truth to attach."""
    files = list_candidate_files()
    subset = stratified_sample(files, n_images, seed=seed)
    for path in subset:
        modality = _modality_for(path)
        for prompt in PROMPTS:
            sample_id = f"{os.path.basename(path)}::{prompt}"
            yield {
                "id": sample_id,
                "images": [{"path": path, "modality": modality, "capture_date": None}],
                "query_text": prompt,
                "meta": {
                    "source_file": os.path.basename(path),
                    "dataset": "local testing/ folder (unlabeled, see module docstring)",
                    "ground_truth_available": False,
                },
            }
