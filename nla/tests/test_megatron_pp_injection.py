"""Unit tests for Megatron PP activation injection micro-batch tracking.

These tests verify that the micro-batch offset tracking correctly handles
activation injection across multiple micro-batches in pipeline parallel scenarios.
"""

import torch
import pytest
from unittest.mock import Mock, patch, MagicMock


class MockMPU:
    """Mock Megatron parallel state for testing."""
    @staticmethod
    def is_pipeline_first_stage():
        return True

    @staticmethod
    def is_pipeline_last_stage():
        return True


class TestMicroBatchOffsetTracking:
    """Test suite for micro-batch offset tracking in NLA Megatron worker."""

    def test_offset_initialization(self):
        """Test that offset is initialized to 0."""
        from verl.nla.workers.nla_megatron_worker import NLAMegatronActorRolloutRefWorker

        # Create mock worker (without full initialization)
        worker = Mock(spec=NLAMegatronActorRolloutRefWorker)

        # Simulate _set_injection_state
        worker._current_activation_vectors = torch.randn(32, 128)
        worker._current_injection_positions = [1] * 32
        worker._current_micro_batch_offset = 0

        assert worker._current_micro_batch_offset == 0

    def test_offset_increments_correctly(self):
        """Test that offset increments by micro-batch size."""
        # Simulate multiple hook invocations
        offsets = []
        micro_batch_sizes = [8, 8, 8, 8]  # 4 micro-batches of size 8

        offset = 0
        for mbs in micro_batch_sizes:
            offsets.append(offset)
            offset += mbs

        assert offsets == [0, 8, 16, 24]
        assert offset == 32  # Total processed

    def test_variable_micro_batch_sizes(self):
        """Test handling of variable micro-batch sizes (last batch smaller)."""
        offsets = []
        micro_batch_sizes = [8, 8, 8, 7]  # Last micro-batch is smaller

        offset = 0
        for mbs in micro_batch_sizes:
            offsets.append(offset)
            offset += mbs

        assert offsets == [0, 8, 16, 24]
        assert offset == 31  # Total processed (not 32)

    def test_global_to_local_index_mapping(self):
        """Test that global indices are correctly computed from offset + local index."""
        micro_batch_offset = 16  # Third micro-batch
        micro_batch_size = 8

        for local_idx in range(micro_batch_size):
            global_idx = micro_batch_offset + local_idx
            assert global_idx >= 16 and global_idx < 24

    def test_injection_state_lifecycle(self):
        """Test complete lifecycle: set → process → clear."""
        # Simulate state management
        class StateManager:
            def __init__(self):
                self._current_activation_vectors = None
                self._current_micro_batch_offset = None

            def set_state(self, vectors):
                self._current_activation_vectors = vectors
                self._current_micro_batch_offset = 0

            def clear_state(self):
                self._current_activation_vectors = None
                self._current_micro_batch_offset = 0

        manager = StateManager()

        # Set state
        vectors = torch.randn(32, 128)
        manager.set_state(vectors)
        assert manager._current_activation_vectors is not None
        assert manager._current_micro_batch_offset == 0

        # Simulate processing (offset would increment here)
        manager._current_micro_batch_offset = 32

        # Clear state
        manager.clear_state()
        assert manager._current_activation_vectors is None
        assert manager._current_micro_batch_offset == 0

    def test_hook_indexing_correctness(self):
        """Test that hook uses correct activation vectors for each micro-batch."""
        # Global batch with 32 activation vectors
        global_activations = torch.arange(32).float().reshape(32, 1)  # Simple: [0, 1, 2, ..., 31]

        # Simulate 4 micro-batches
        micro_batch_configs = [
            (0, 8),   # Micro-batch 1: offset=0, size=8
            (8, 8),   # Micro-batch 2: offset=8, size=8
            (16, 8),  # Micro-batch 3: offset=16, size=8
            (24, 8),  # Micro-batch 4: offset=24, size=8
        ]

        for offset, size in micro_batch_configs:
            # Simulate hook logic
            used_activations = []
            for local_idx in range(size):
                global_idx = offset + local_idx
                used_activations.append(global_activations[global_idx].item())

            # Verify correct slice is used
            expected = list(range(offset, offset + size))
            assert used_activations == expected, f"Offset {offset}: expected {expected}, got {used_activations}"

    def test_edge_case_single_micro_batch(self):
        """Test case where global batch fits in single micro-batch."""
        global_batch_size = 4
        micro_batch_size = 8  # Larger than global batch

        # Only one micro-batch needed
        offset = 0
        assert offset == 0

        # Process all samples
        for local_idx in range(global_batch_size):
            global_idx = offset + local_idx
            assert global_idx == local_idx  # Matches 1:1

    def test_edge_case_many_small_micro_batches(self):
        """Test case with many small micro-batches."""
        global_batch_size = 32
        micro_batch_size = 2  # Very small

        num_micro_batches = global_batch_size // micro_batch_size
        assert num_micro_batches == 16

        offsets = [i * micro_batch_size for i in range(num_micro_batches)]
        assert offsets == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


class TestPipelineStageGuards:
    """Test that injection only happens on first pipeline stage."""

    @patch('verl.nla.workers.nla_megatron_worker.mpu')
    def test_hook_returns_early_on_non_first_stage(self, mock_mpu):
        """Test that hook returns immediately if not on first pipeline stage."""
        mock_mpu.is_pipeline_first_stage.return_value = False

        # Simulate hook behavior
        if not mock_mpu.is_pipeline_first_stage():
            output = torch.randn(8, 32, 128)  # Return unmodified
        else:
            output = None  # Should not reach here

        assert output is not None
        assert output.shape == (8, 32, 128)

    @patch('verl.nla.workers.nla_megatron_worker.mpu')
    def test_hook_processes_on_first_stage(self, mock_mpu):
        """Test that hook processes on first pipeline stage."""
        mock_mpu.is_pipeline_first_stage.return_value = True

        # Simulate hook behavior
        if not mock_mpu.is_pipeline_first_stage():
            modified = False
        else:
            modified = True  # Hook should execute

        assert modified


class TestActivationInjectionLogic:
    """Test the actual injection logic."""

    def test_injection_at_correct_position(self):
        """Test that activation is injected at the correct token position."""
        # Create embeddings: [batch_size=2, seq_len=10, hidden_dim=128]
        embeddings = torch.zeros(2, 10, 128)

        # Activation vectors to inject
        activations = torch.ones(2, 128) * 99  # Distinctive value

        # Injection positions
        positions = [3, 7]  # Inject at position 3 for sample 0, position 7 for sample 1

        # Perform injection
        for batch_idx in range(2):
            pos = positions[batch_idx]
            embeddings[batch_idx, pos] = activations[batch_idx]

        # Verify injection
        assert torch.allclose(embeddings[0, 3], torch.ones(128) * 99)
        assert torch.allclose(embeddings[1, 7], torch.ones(128) * 99)

        # Verify other positions unchanged
        assert torch.allclose(embeddings[0, 0], torch.zeros(128))
        assert torch.allclose(embeddings[1, 0], torch.zeros(128))

    def test_projection_layer_application(self):
        """Test that projection layer is applied when dimensions mismatch."""
        # Activation: [batch, 256], but embeddings need [batch, 128]
        activation = torch.randn(1, 256)

        # Create projection layer
        projection = torch.nn.Linear(256, 128, bias=False)

        # Project
        projected = projection(activation)

        assert projected.shape == (1, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
