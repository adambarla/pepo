"""Tests for device manager GPU assignment and error handling."""

from unittest.mock import patch

import torch

from pepo.utils.device import DeviceManager


class TestRequestGpuErrorSwallowing:
    """request_gpu swallows sync errors, corrupting CUDA context."""

    def test_sync_error_is_swallowed(self):
        """Demonstrate that CUDA sync errors in request_gpu are silently ignored."""
        dm = DeviceManager(gpu_ids=[0])
        called_synchronize = False

        def _raise_on_sync():
            nonlocal called_synchronize
            called_synchronize = True
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        with patch.object(torch.cuda, "synchronize", side_effect=_raise_on_sync):
            with dm.request_gpu() as device:
                assert str(device) == "cuda:0"

        assert called_synchronize, "synchronize was never called"

    def test_sync_error_is_invisible_to_caller(self):
        """Swallowed error is invisible: no exception, but the context is corrupted."""
        dm = DeviceManager(gpu_ids=[0])
        errors_caught = []

        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("CUDA error")
        ):
            try:
                with dm.request_gpu() as device:
                    assert str(device) == "cuda:0"
            except RuntimeError as e:
                errors_caught.append(e)

        assert len(errors_caught) == 0, (
            "ERROR was propagated through request_gpu despite being swallowed"
        )

    def test_gpu_returned_after_error_swallow(self):
        """GPU returns to the pool after a sync error, allowing reuse when corrupted."""
        dm = DeviceManager(gpu_ids=[0])

        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("CUDA error")
        ):
            with dm.request_gpu():
                pass

        # GPU 0 returned — next caller gets it with corrupted context
        next_gpu = dm._gpu_queue.get(timeout=1)
        assert next_gpu == 0

    def test_concurrent_request_gpu_fairness(self):
        """request_gpu round-robins GPUs — a single caller gets sequential GPUs."""
        dm = DeviceManager(gpu_ids=[0, 1, 2, 3])
        assigned = []
        for _ in range(6):
            with dm.request_gpu() as d:
                assigned.append(str(d))
        # With 4 GPUs, 6 requests, the last 2 repeat
        assert assigned[:4] == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        assert assigned[4:] == ["cuda:0", "cuda:1"]

    def test_gpu_migration_between_epochs(self):
        """Simulate per-epoch request_gpu — model gets different GPUs each time."""
        dm = DeviceManager(gpu_ids=[0, 1])
        gpus = []

        class DummyModel:
            def to(self, device):
                gpus.append(str(device))

            device = None

        model = DummyModel()

        for epoch in range(3):
            with dm.request_gpu() as device:
                model.to(device)

        # Model moved to different GPUs each epoch
        assert gpus == ["cuda:0", "cuda:1", "cuda:0"], f"Got: {gpus}"

    def test_shared_backbone_cpu_fallback_not_needed(self):
        """With shared_backbone=False, model.to() moves full base, not just adapters.
        This test demonstrates that per-epoch GPU reassignment is unnecessary
        since each model already has its own GPU via static assignment."""
        dm = DeviceManager(gpu_ids=[0, 1, 2, 3])
        for model_idx in range(4):
            gpu_id = dm._get_gpu_id_for_model(model_idx)
            assert gpu_id == model_idx, f"Model {model_idx} got GPU {gpu_id}"
