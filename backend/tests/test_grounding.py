"""Tests for the deterministic grounding specialist."""

import os

import cv2
import numpy as np
import pytest

from app.services.grounding import ground_object, SUPPORTED_TARGETS


# ── Helpers ───────────────────────────────────────────────────────────

def _make_hsv_image(path: str, h: int, s: int, v: int, size: int = 200):
    """Create a solid-colour image via HSV → BGR and save it."""
    hsv = np.full((size, size, 3), (h, s, v), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, bgr)


def _make_image_with_region_hsv(
    path: str,
    bg_hsv: tuple[int, int, int],
    region_hsv: tuple[int, int, int],
    size: int = 200,
    region: tuple[int, int, int, int] = (40, 40, 80, 80),
):
    """Create an image with a rectangular region of a different HSV colour."""
    hsv = np.full((size, size, 3), bg_hsv, dtype=np.uint8)
    x, y, w, h = region
    hsv[y : y + h, x : x + w] = region_hsv
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path, bgr)


# ── Tests ─────────────────────────────────────────────────────────────

class TestWaterGrounding:

    def test_detects_blue_water_region(self, tmp_path):
        p = str(tmp_path / "water.png")
        # Blue water: hue ~100, high sat, moderate value
        _make_image_with_region_hsv(p, bg_hsv=(30, 50, 150), region_hsv=(100, 180, 180))

        result = ground_object(p, "water")
        assert len(result["bounding_boxes"]) >= 1, "Should detect the blue region"
        assert result["confidence_score"] is not None
        assert result["confidence_score"] > 0
        assert "water" in result["answer_text"].lower()

    def test_no_water_in_green_image(self, tmp_path):
        p = str(tmp_path / "green.png")
        _make_hsv_image(p, h=50, s=150, v=150)

        result = ground_object(p, "water")
        assert len(result["bounding_boxes"]) == 0
        assert "No water" in result["answer_text"]


class TestVegetationGrounding:

    def test_detects_green_vegetation(self, tmp_path):
        p = str(tmp_path / "veg.png")
        # Green vegetation: hue ~50, high sat, moderate value
        _make_image_with_region_hsv(p, bg_hsv=(10, 80, 120), region_hsv=(50, 180, 160))

        result = ground_object(p, "vegetation")
        assert len(result["bounding_boxes"]) >= 1, "Should detect the green region"
        assert result["confidence_score"] is not None

    def test_no_vegetation_in_blue_image(self, tmp_path):
        p = str(tmp_path / "blue.png")
        _make_hsv_image(p, h=100, s=150, v=150)

        result = ground_object(p, "vegetation")
        assert len(result["bounding_boxes"]) == 0


class TestBuiltUpGrounding:

    def test_unsupported_target_fails_gracefully(self, tmp_path):
        p = str(tmp_path / "any.png")
        _make_hsv_image(p, h=100, s=50, v=150)

        result = ground_object(p, "buildings_xyz")
        assert result["bounding_boxes"] == []
        assert result["confidence_score"] is None
        assert "Unsupported" in result["answer_text"]


class TestEdgeCases:

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            ground_object("/nonexistent/image.png", "water")

    def test_supported_targets_list(self):
        assert "water" in SUPPORTED_TARGETS
        assert "vegetation" in SUPPORTED_TARGETS
        assert "built-up" in SUPPORTED_TARGETS

    def test_bounding_box_format(self, tmp_path):
        p = str(tmp_path / "water.png")
        _make_image_with_region_hsv(p, bg_hsv=(30, 50, 100), region_hsv=(100, 200, 200), size=300, region=(50, 50, 100, 100))

        result = ground_object(p, "water")
        for box in result["bounding_boxes"]:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1 and y2 > y1
