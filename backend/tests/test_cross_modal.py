"""Focused unit tests for the deterministic optical + SAR cross-modal service."""

import io

from PIL import Image as PILImage

from app.services.cross_modal import analyze_pair, TOOL_VERSION


def _write_png(path, color, size=64, grayscale=False):
    if grayscale:
        img = PILImage.new("L", (size, size), color=color)
    else:
        img = PILImage.new("RGB", (size, size), color=color)
    img.save(path, format="PNG")
    return str(path)


def test_analyze_pair_returns_structured_output(tmp_path):
    optical_path = _write_png(tmp_path / "optical.png", (0, 200, 0))  # strong green
    sar_path = _write_png(tmp_path / "sar.png", 20, grayscale=True)   # very dark backscatter

    result = analyze_pair(optical_path, sar_path, query_text="Analyze this pair")

    assert result["answer_text"]
    assert result["confidence_score"] is None
    assert result["model_version"] == TOOL_VERSION
    assert result["execution_status"] == "completed"

    evidence = result["evidence"]
    assert set(evidence) == {"optical", "sar", "combined"}

    optical = evidence["optical"]
    assert optical["vegetation_green_dominance"] == 1.0
    assert optical["water_blue_dominance"] == 0.0
    assert optical["brightness_0_255"] > 0
    assert optical["total_pixels"] == 64 * 64

    sar = evidence["sar"]
    assert sar["dark_low_backscatter_fraction"] == 1.0
    assert sar["bright_high_backscatter_fraction"] == 0.0
    assert sar["mean_backscatter_0_255"] == 20.0
    assert sar["total_pixels"] == 64 * 64

    assert evidence["combined"]["n_readings"] >= 1
    assert "modality_contribution_note" in result


def test_analyze_pair_water_signature(tmp_path):
    # Blue-dominant optical + dark SAR -> water-like agreement
    optical_path = _write_png(tmp_path / "opt_water.png", (20, 40, 180))
    sar_path = _write_png(tmp_path / "sar_water.png", 10, grayscale=True)

    result = analyze_pair(optical_path, sar_path)
    combined_text = " ".join(result["evidence"]["combined"]["combined_readings"]).lower()
    assert result["model_version"] == TOOL_VERSION
    assert result["evidence"]["optical"]["water_blue_dominance"] > 0.5
    assert "agree on likely water" in combined_text


def test_analyze_pair_does_not_fabricate_confidence(tmp_path):
    optical_path = _write_png(tmp_path / "o.png", (60, 60, 60))
    sar_path = _write_png(tmp_path / "s.png", 128, grayscale=True)

    result = analyze_pair(optical_path, sar_path)
    assert result["confidence_score"] is None


def test_sar_regions_are_capped_for_overlay_readability(tmp_path):
    # Many separate high-backscatter blobs must be capped so the overlay
    # never shows 18+ overlapping labels.
    img = PILImage.new("L", (128, 128), color=80)
    draw = img.load()
    for i in range(4):
        for j in range(4):
            x, y = 10 + i * 27, 10 + j * 27
            for yy in range(y, y + 12):
                for xx in range(x, x + 12):
                    draw[xx, yy] = 255
    sar_path = str(tmp_path / "blobs.png")
    img.save(sar_path, format="PNG")

    optical_path = _write_png(tmp_path / "o.png", (60, 60, 60))
    result = analyze_pair(optical_path, sar_path)
    sar_regions = result["evidence"]["sar"]["regions"] or []
    assert sar_regions, "Bright blobs must be detected"
    assert len(sar_regions) <= 8, "Overlay must not be cluttered with 18+ boxes"