"""Tests for NLA data utilities."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from verl.nla.data import NLADataset, NLABatch, NLADataCollator


class TestNLADataset:
    """Test suite for NLADataset."""

    def setup_method(self):
        """Setup mock tokenizer for tests."""
        self.tokenizer = Mock()
        self.tokenizer.get_vocab.return_value = {}
        self.tokenizer.add_special_tokens = Mock()
        self.tokenizer.convert_tokens_to_ids = Mock(return_value=100)
        self.tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10),
        }

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        activation_vectors = torch.randn(2, 768)

        dataset = NLADataset(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=self.tokenizer,
        )

        assert len(dataset) == 2
        assert dataset.injection_token == "<INJECT>"
        assert isinstance(dataset.activation_vectors, torch.Tensor)

    def test_dataset_with_numpy_vectors(self):
        """Test dataset with numpy activation vectors."""
        prompts = ["Test prompt"]
        activation_vectors = np.random.randn(1, 768)

        dataset = NLADataset(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=self.tokenizer,
        )

        assert isinstance(dataset.activation_vectors, torch.Tensor)
        assert dataset.activation_vectors.shape == (1, 768)

    def test_dataset_with_list_vectors(self):
        """Test dataset with list of activation vectors."""
        prompts = ["Test 1", "Test 2"]
        activation_vectors = [
            np.random.randn(768),
            np.random.randn(768),
        ]

        dataset = NLADataset(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=self.tokenizer,
        )

        assert len(dataset.activation_vectors) == 2
        assert all(isinstance(v, torch.Tensor) for v in dataset.activation_vectors)

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        prompts = ["Test with <INJECT> token"]
        activation_vectors = torch.randn(1, 768)

        # Setup tokenizer to return injection token
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 100, 4, 5]]),
            "attention_mask": torch.ones(1, 5),
        }

        dataset = NLADataset(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=self.tokenizer,
            max_length=5,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "activation_vector" in item
        assert "injection_positions" in item
        assert item["injection_positions"] == [2]  # Position of token 100

    def test_dataset_with_labels(self):
        """Test dataset with labels."""
        prompts = ["Test 1", "Test 2"]
        activation_vectors = torch.randn(2, 768)
        labels = [0, 1]

        dataset = NLADataset(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=self.tokenizer,
            labels=labels,
        )

        item = dataset[0]
        assert "labels" in item
        assert item["labels"] == 0


class TestNLADataCollator:
    """Test suite for NLADataCollator."""

    def test_basic_collation(self):
        """Test basic batch collation."""
        collator = NLADataCollator()

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": torch.ones(5),
                "activation_vector": torch.randn(768),
                "injection_positions": [2],
            },
            {
                "input_ids": torch.tensor([6, 7, 8, 9, 10]),
                "attention_mask": torch.ones(5),
                "activation_vector": torch.randn(768),
                "injection_positions": [0, 4],
            },
        ]

        batch = collator(features)

        assert batch["input_ids"].shape == (2, 5)
        assert batch["attention_mask"].shape == (2, 5)
        assert batch["activation_vectors"].shape == (2, 768)
        assert len(batch["injection_positions"]) == 2

    def test_variable_size_activation_padding(self):
        """Test padding of variable-size activation vectors."""
        collator = NLADataCollator()

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "activation_vector": torch.randn(512),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "activation_vector": torch.randn(768),
            },
        ]

        batch = collator(features)

        # Should pad to max dimension (768)
        assert batch["activation_vectors"].shape == (2, 768)

    def test_rl_fields_collation(self):
        """Test collation of RL-specific fields."""
        collator = NLADataCollator()

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "response_ids": torch.tensor([4, 5, 6]),
                "response_mask": torch.ones(3),
                "reward": torch.tensor(1.0),
            },
            {
                "input_ids": torch.tensor([7, 8, 9]),
                "response_ids": torch.tensor([10, 11, 12]),
                "response_mask": torch.ones(3),
                "reward": torch.tensor(2.0),
            },
        ]

        batch = collator(features)

        assert "response_ids" in batch
        assert "response_mask" in batch
        assert "rewards" in batch
        assert batch["rewards"].shape == (2,)


class TestNLABatch:
    """Test suite for NLABatch dataclass."""

    def test_batch_creation(self):
        """Test creating NLABatch."""
        batch = NLABatch(
            input_ids=torch.randint(0, 1000, (2, 10)),
            activation_vectors=torch.randn(2, 768),
            attention_mask=torch.ones(2, 10),
        )

        assert batch.input_ids.shape == (2, 10)
        assert batch.activation_vectors.shape == (2, 768)
        assert batch.attention_mask.shape == (2, 10)
        assert batch.labels is None
        assert batch.metadata is None


if __name__ == "__main__":
    # Run dataset tests
    dataset_tests = TestNLADataset()
    dataset_tests.setup_method()
    dataset_tests.test_dataset_initialization()
    dataset_tests.test_dataset_with_numpy_vectors()
    dataset_tests.test_dataset_with_list_vectors()
    dataset_tests.test_dataset_getitem()
    dataset_tests.test_dataset_with_labels()

    # Run collator tests
    collator_tests = TestNLADataCollator()
    collator_tests.test_basic_collation()
    collator_tests.test_variable_size_activation_padding()
    collator_tests.test_rl_fields_collation()

    # Run batch tests
    batch_tests = TestNLABatch()
    batch_tests.test_batch_creation()

    print("All data utility tests passed!")