"""Tests for the deterministic change-detection specialist."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from app.services.change_detection import detect_change


# ── Helpers ───────────────────────────────────────────────────────────

def _make_image(path: str, color: tuple[int, int, int] = (100, 100, 100), size: int = 100):
    """Write a solid-colour BGR image to *path*."""
    img = np.full((size, size, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_image_with_region(
    path: str,
    bg_color: tuple[int, int, int] = (100, 100, 100),
    region_color: tuple[int, int, int] = (255, 255, 255),
    size: int = 100,
    region: tuple[int, int, int, int] = (20, 20, 50, 50),
):
    """Write an image with a solid background and a contrasting rectangular region."""
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    x, y, w, h = region
    img[y : y + h, x : x + w] = region_color
    cv2.imwrite(path, img)


# ── Tests ─────────────────────────────────────────────────────────────

class TestChangeDetection:

    def test_identical_images_no_change(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(80, 90, 100))
        _make_image(p2, color=(80, 90, 100))

        result = detect_change(p1, p2)
        assert result["change_percentage"] == 0.0
        assert result["num_regions"] == 0
        assert result["bounding_boxes"] == []
        assert result["change_mask_path"] is None
        assert "were not detected" in result["answer_text"]

    def test_different_images_detect_change(self, tmp_path):
        p1 = str(tmp_path / "before.png")
        p2 = str(tmp_path / "after.png")
        _make_image(p1, color=(50, 50, 50))
        # Large bright region in the 'after' image
        _make_image_with_region(p2, bg_color=(50, 50, 50), region_color=(255, 255, 255), region=(10, 10, 40, 40))

        result = detect_change(p1, p2)
        assert result["change_percentage"] > 0
        assert result["num_regions"] >= 1
        assert len(result["bounding_boxes"]) >= 1
        assert result["change_mask_path"] is not None
        assert os.path.isfile(result["change_mask_path"])

    def test_bounding_box_has_correct_shape(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(30, 30, 30))
        _make_image_with_region(p2, bg_color=(30, 30, 30), region_color=(200, 200, 200), region=(0, 0, 50, 50))

        result = detect_change(p1, p2)
        for box in result["bounding_boxes"]:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1
            assert y2 > y1

    def test_different_dimensions_resized(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        # Different sizes — should resize without crashing
        img1 = np.full((80, 120, 3), (60, 60, 60), dtype=np.uint8)
        img2 = np.full((60, 100, 3), (60, 60, 60), dtype=np.uint8)
        cv2.imwrite(p1, img1)
        cv2.imwrite(p2, img2)

        result = detect_change(p1, p2)
        assert result["change_percentage"] == 0.0

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            detect_change("/nonexistent/a.png", "/nonexistent/b.png")

    def test_mask_saved_when_change_exists(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(40, 40, 40))
        _make_image_with_region(p2, bg_color=(40, 40, 40), region_color=(220, 220, 220), region=(0, 0, 60, 60))

        result = detect_change(p1, p2)
        if result["bounding_boxes"]:
            assert result["change_mask_path"] is not None
            assert os.path.getsize(result["change_mask_path"]) > 0


class TestChangeValidation:
    """The service must refuse to invent percentages for non-comparable pairs."""

    def test_incompatible_aspect_ratio_rejected(self, tmp_path):
        p1 = str(tmp_path / "wide.png")
        p2 = str(tmp_path / "square.png")
        cv2.imwrite(p1, np.full((40, 200, 3), (80, 80, 80), dtype=np.uint8))   # aspect 5.0
        cv2.imwrite(p2, np.full((120, 120, 3), (80, 80, 80), dtype=np.uint8))  # aspect 1.0

        result = detect_change(p1, p2)
        assert result["validation_failed"] is True
        assert result["error"] is True
        assert result["change_percentage"] is None
        assert result["bounding_boxes"] is None
        assert result["change_mask_path"] is None
        assert "spatial correspondence" in result["answer_text"]

    def test_incompatible_bbox_metadata_rejected(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1)
        _make_image(p2)

        meta_before = {"crs": "EPSG:32635", "bbox_coords": [0, 0, 100, 100]}
        meta_after = {"crs": "EPSG:32635", "bbox_coords": [1000, 1000, 1100, 1100]}
        result = detect_change(p1, p2, metadata_before=meta_before, metadata_after=meta_after)
        assert result["validation_failed"] is True
        assert "spatial correspondence" in result["reason"]

    def test_near_full_frame_diff_not_reported_as_change(self, tmp_path):
        # Whole-frame difference (e.g. neighbouring scenes, spurious periodic
        # registration peak) must be refused, not reported as "~99% changed".
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(0, 0, 0))
        _make_image(p2, color=(255, 255, 255))

        result = detect_change(p1, p2)
        assert result["validation_failed"] is True
        assert result["error"] is True
        assert result["change_percentage"] is None
        assert result["bounding_boxes"] is None
        assert result["overlay_path"] is None
        assert "spatial correspondence could not be verified" in result["answer_text"]

    def test_same_tile_metadata_accepted(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(60, 60, 60))
        _make_image_with_region(p2, bg_color=(60, 60, 60), region_color=(220, 220, 220), region=(10, 10, 30, 30))
        meta = {"tile_id": "T35ULA"}
        result = detect_change(p1, p2, metadata_before=meta, metadata_after=meta)
        assert result["validation_failed"] is False

    def test_speckle_noise_does_not_create_region(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        rng = np.random.default_rng(7)
        img1 = np.full((120, 120, 3), (90, 90, 90), dtype=np.uint8)
        img2 = img1.copy()
        ys = rng.integers(0, 120, size=8)
        xs = rng.integers(0, 120, size=8)
        img2[ys, xs] = 255
        cv2.imwrite(p1, img1)
        cv2.imwrite(p2, img2)

        result = detect_change(p1, p2)
        # isolated single-pixel specks must be removed by morphological opening
        assert result["num_regions"] == 0

    def test_multiple_clear_regions_are_separated(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(40, 40, 40))
        img = np.full((100, 100, 3), (40, 40, 40), dtype=np.uint8)
        img[10:30, 10:30] = (255, 255, 255)
        img[70:90, 70:90] = (255, 255, 255)
        cv2.imwrite(p2, img)

        result = detect_change(p1, p2)
        assert result["num_regions"] >= 2
        assert len(result["bounding_boxes"]) >= 2
        # far-apart regions must not merge into one giant box
        areas = sorted(
            (b[2] - b[0]) * (b[3] - b[1]) for b in result["bounding_boxes"]
        )
        assert areas[0] < 60 * 60

    def test_overlay_is_saved_with_change(self, tmp_path):
        p1 = str(tmp_path / "a.png")
        p2 = str(tmp_path / "b.png")
        _make_image(p1, color=(50, 50, 50))
        _make_image_with_region(p2, bg_color=(50, 50, 50), region_color=(255, 255, 255), region=(10, 10, 40, 40))

        result = detect_change(p1, p2)
        if result["bounding_boxes"] and result["overlay_path"]:
            assert os.path.isfile(result["overlay_path"])
            ov = cv2.imread(result["overlay_path"])
            assert ov is not None
