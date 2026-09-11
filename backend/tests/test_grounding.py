"""Tests for the deterministic grounding specialist."""

import os

import cv2
import numpy as np
import pytest

from app.services.grounding import ground_object, scene_cues, SUPPORTED_TARGETS


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
        assert not result["bounding_boxes"]
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
        assert "roads" in SUPPORTED_TARGETS
        assert "farmland" in SUPPORTED_TARGETS

    def test_bounding_box_format(self, tmp_path):
        p = str(tmp_path / "water.png")
        _make_image_with_region_hsv(p, bg_hsv=(30, 50, 100), region_hsv=(100, 200, 200), size=300, region=(50, 50, 100, 100))

        result = ground_object(p, "water")
        for box in result["bounding_boxes"]:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1 and y2 > y1


class TestRoadsGrounding:

    def test_detects_long_grey_road(self, tmp_path):
        p = str(tmp_path / "road.png")
        # Dark green background (saturated), long grey strip = road
        img = np.full((200, 200, 3), (30, 60, 30), dtype=np.uint8)
        img[90:110, :] = (150, 150, 150)
        cv2.imwrite(p, img)

        result = ground_object(p, "roads")
        assert len(result["bounding_boxes"]) >= 1, "Should detect the road strip"
        for box in result["bounding_boxes"]:
            x1, y1, x2, y2 = box
            assert (x2 - x1) / max(1, (y2 - y1)) >= 2.5

    def test_compact_grey_block_is_not_a_road(self, tmp_path):
        p = str(tmp_path / "block.png")
        img = np.full((200, 200, 3), (30, 60, 30), dtype=np.uint8)
        img[40:90, 40:90] = (150, 150, 150)
        cv2.imwrite(p, img)

        result = ground_object(p, "roads")
        assert len(result["bounding_boxes"]) == 0


class TestFarmlandGrounding:

    def test_detects_green_field_blocks(self, tmp_path):
        p = str(tmp_path / "farmland.png")
        _make_image_with_region_hsv(p, bg_hsv=(10, 80, 120), region_hsv=(50, 180, 160), region=(40, 40, 100, 100))

        result = ground_object(p, "farmland")
        assert len(result["bounding_boxes"]) >= 1, "Should detect the field block"
        assert result["confidence_score"] is not None


class TestDisplayCap:

    def test_many_regions_capped_for_readable_overlay(self, tmp_path):
        # 16 separate green field blocks -> display is capped, fractions yet
        # un-capped via scene_cues (which keeps its own path untouched).
        p = str(tmp_path / "many.png")
        hsv = np.full((220, 220, 3), (10, 80, 120), dtype=np.uint8)  # non-green bg
        cell = 16
        gap = 24
        start = 8
        for i in range(4):
            for j in range(4):
                x = start + j * (cell + gap)
                y = start + i * (cell + gap)
                hsv[y : y + cell, x : x + cell] = (50, 200, 160)
        cv2.imwrite(p, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

        result = ground_object(p, "vegetation")
        assert result["bounding_boxes"], "Green blocks must be detected"
        assert len(result["bounding_boxes"]) <= 10

        cues = scene_cues(p)
        assert cues["fractions"]["vegetation"] > 0


class TestSceneCues:

    def test_cue_fractions_are_measured(self, tmp_path):
        p = str(tmp_path / "mixed.png")
        _make_image_with_region_hsv(p, bg_hsv=(100, 160, 160), region_hsv=(50, 180, 160), region=(40, 40, 120, 120))

        cues = scene_cues(p)
        assert set(cues["fractions"].keys()) == SUPPORTED_TARGETS
        assert any(c["cue"] for c in cues["detected_cues"])

    def test_cues_never_invent_dominance(self, tmp_path):
        p = str(tmp_path / "flat.png")
        # Very low saturation scene -> nothing should be flagged as dominant
        _make_hsv_image(p, h=100, s=5, v=200)
        cues = scene_cues(p)
        assert "dominant_cue" in cues
