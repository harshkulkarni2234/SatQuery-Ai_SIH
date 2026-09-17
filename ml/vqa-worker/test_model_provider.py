"""Unit test for SmolVLMProvider's device selection (Phase B2/B8 GPU
re-check). Deliberately does not load the real model — that's covered by
manually running the worker (see docs/DEMO_RUNBOOK.md) — this only locks
in the CUDA > MPS > CPU priority so a future change can't silently regress
back to skipping MPS, which is exactly the bug this fixed.

Run with: ml/vqa-worker/venv/bin/python -m pytest ml/vqa-worker/test_model_provider.py -q
"""

from unittest.mock import MagicMock, patch

from model_provider import SmolVLMProvider


def _make_provider_without_load():
    # __init__ only sets self.device/self.dtype; doesn't touch the network,
    # so we can construct it directly and inspect the result.
    return SmolVLMProvider(model_dir="unused", adapter_dir="unused")


def test_prefers_cuda_when_available():
    with patch("model_provider.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        provider = _make_provider_without_load()
        assert provider.device == "cuda"
        assert provider.dtype == "float16"


def test_uses_mps_when_no_cuda_but_mps_available():
    with patch("model_provider.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        provider = _make_provider_without_load()
        assert provider.device == "mps"
        # fp16 is deliberately NOT used on MPS (see model_provider.py comment)
        assert provider.dtype == "float32"


def test_falls_back_to_cpu_when_neither_available():
    with patch("model_provider.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        provider = _make_provider_without_load()
        assert provider.device == "cpu"
        assert provider.dtype == "float32"
