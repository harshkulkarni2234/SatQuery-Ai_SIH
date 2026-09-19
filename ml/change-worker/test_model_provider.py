"""Unit test for ChangeModelProvider's device selection.

Deliberately does not load the real model weights — that's covered
by manually running the worker (see docs/DEMO_RUNBOOK.md). This
only locks in the CUDA > CPU priority so a future change can't
silently regress back to skipping the GPU, which is exactly the
bug the vqa-worker's test_model_provider.py fixed.

Run with: ml/change-worker/venv/bin/python -m pytest ml/change-worker/test_model_provider.py -q
"""

from unittest.mock import MagicMock, patch

from model_provider import ChangeModelProvider


def _make_provider_without_load():
    return ChangeModelProvider()


def test_prefers_cuda_when_available():
    with patch("model_provider.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        provider = _make_provider_without_load()
        assert provider.device == "cuda"
        assert provider.dtype == "float16"


def test_falls_back_to_cpu_when_no_cuda():
    with patch("model_provider.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        provider = _make_provider_without_load()
        assert provider.device == "cpu"
        assert provider.dtype == "float32"
