"""NLA SFT Dataset with activation vector support."""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from transformers import PreTrainedTokenizer

from verl.utils.dataset.sft_dataset import SFTDataset
from verl.nla.utils.injection_manager import InjectionTokenManager
from verl.utils.model import compute_position_id_with_mask


class NLASFTDataset(SFTDataset):
    """
    SFT Dataset for NLA that supports activation vectors.

    This dataset can be used for both:
    1. Actor training: Provides prompts + activation vectors for injection
    2. Critic training: Provides responses + target activation vectors for reconstruction

    Expected parquet columns:
    - prompt: The input prompt text
    - response: The response text
    - activation_vector: The activation vector (list or numpy array)
    - Additional columns as needed by base SFTDataset
    """

    def __init__(
        self,
        parquet_files: Union[str, ListConfig],
        tokenizer: Union[str, PreTrainedTokenizer],
        config: DictConfig,
        mode: str = "actor",  # "actor", "critic", or "both"
    ):
        """
        Initialize NLA SFT dataset.

        Args:
            parquet_files: Path(s) to parquet files
            tokenizer: Tokenizer instance or path
            config: Configuration dict with dataset parameters
            mode: Training mode - "actor", "critic", or "both"
        """
        self.mode = mode
        # Get activation_dim from config - will be determined dynamically from model config
        self.activation_dim = config.get("activation_dim")
        if self.activation_dim is None:
            raise ValueError("activation_dim must be provided in config or determined from model config")

        # Initialize base dataset first to get tokenizer
        super().__init__(parquet_files, tokenizer, config)

        # Set up injection token management
        injection_token = config.get("injection_token", None)
        self.injection_manager = InjectionTokenManager(self.tokenizer, injection_token)
        self.injection_token = self.injection_manager.character
        self.injection_token_id = self.injection_manager.token_id

        # Load activation vectors
        self._load_activation_vectors()

    def _load_activation_vectors(self):
        """Load activation vectors from the dataframe."""
        # Activation vectors should be in the dataframe
        if "activation_vector" not in self.dataframe.columns:
            raise ValueError("Dataset must contain 'activation_vector' column")

        # Convert activation vectors to tensors
        self.activation_vectors = []
        for idx in range(len(self.dataframe)):
            vec = self.dataframe.iloc[idx]["activation_vector"]

            # Handle different formats (list, numpy array, etc.)
            if isinstance(vec, (list, tuple)):
                vec = torch.tensor(vec, dtype=torch.float32)
            elif hasattr(vec, "numpy"):  # pandas/numpy array
                vec = torch.tensor(vec.numpy(), dtype=torch.float32)
            elif isinstance(vec, torch.Tensor):
                vec = vec.float()
            else:
                vec = torch.tensor(vec, dtype=torch.float32)

            # Ensure correct dimension
            if vec.shape[-1] != self.activation_dim:
                raise ValueError(f"Activation vector dimension mismatch: expected {self.activation_dim}, got {vec.shape[-1]}")

            self.activation_vectors.append(vec)

    def _prepare_actor_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepare sample for actor training with activation injection.

        The prompt should contain the injection token where the activation
        vector should be injected.
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)

        # Attach activation vector
        sample["activation_vectors"] = self.activation_vectors[idx]

        if self.injection_token_id not in sample["input_ids"]:
            raise ValueError(
                "Prompt does not contain the expected injection token. "
                "Ensure the dataset prompt_text includes the injection marker."
            )

        # Skip metadata attachment for now to avoid TensorDict batch issues
        # sample = self._attach_metadata(idx, sample)

        return sample

    def _prepare_critic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepare sample for critic training.

        The critic needs:
        - response_ids: The generated response tokens
        - response_attention_mask: Attention mask for the response
        - target_activations: The target activation vector to predict
        """
        # Get the raw prompt and response
        prompt = self.prompts[idx]
        response = self.responses[idx]

        # Tokenize response only
        response_with_eos = response + self.tokenizer.eos_token
        response_output = self.tokenizer(
            response_with_eos,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False
        )

        sample = {
            "response_ids": response_output["input_ids"][0],
            "response_attention_mask": response_output["attention_mask"][0],
            "activation_vectors": self.activation_vectors[idx],  # Target for critic
        }

        # Also include full sequence for potential joint training
        full_sample = super().__getitem__(idx)
        sample.update({
            "input_ids": full_sample["input_ids"],
            "attention_mask": full_sample["attention_mask"],
            "loss_mask": full_sample["loss_mask"],
        })

        # Skip metadata attachment for now to avoid TensorDict batch issues
        # sample = self._attach_metadata(idx, sample)
        return sample

    def _attach_metadata(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        row = self.dataframe.iloc[idx]

        metadata_keys = [
            "sample_uuid",
            "source",
            "formatted_source",
            "tokenized_source",
            "forward_pass_text",
            "prompt_text",
            "activation_layer",
            "activation_token",
            "activation_token_id",
            "source_message_token_index",
            "source_sequence_token_index",
            "prompt_token_index",
            "source_messages",
        ]

        extra_info = sample.get("extra_info", {}) or {}
        for key in metadata_keys:
            if key in row and pd.notna(row[key]):
                value = row[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                extra_info[key] = value

        extra_info["response"] = self.responses[idx]
        sample["extra_info"] = extra_info

        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns different formats based on training mode:
        - actor: Sample with injection token and activation vector
        - critic: Sample with response and target activation
        - both: Combined sample with all fields
        """
        if self.mode == "actor":
            return self._prepare_actor_sample(idx)
        elif self.mode == "critic":
            return self._prepare_critic_sample(idx)
        elif self.mode == "both":
            # Combine both actor and critic samples
            actor_sample = self._prepare_actor_sample(idx)
            critic_sample = self._prepare_critic_sample(idx)

            # Merge samples, with critic fields prefixed
            combined_sample = actor_sample.copy()
            combined_sample.update({
                "response_ids": critic_sample["response_ids"],
                "response_attention_mask": critic_sample["response_attention_mask"],
            })
            return combined_sample
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class NLASFTCollator:
    """
    Data collator for NLA SFT that properly handles activation vectors.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples, handling activation vectors properly.
        """
        batch = {}

        # Stack standard tensor fields
        tensor_keys = ["input_ids", "attention_mask", "loss_mask", "position_ids"]
        for key in tensor_keys:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Stack response fields if present
        response_keys = ["response_ids", "response_attention_mask"]
        for key in response_keys:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Stack activation vectors
        if "activation_vectors" in features[0]:
            # Ensure all activation vectors have the same shape
            batch["activation_vectors"] = torch.stack([f["activation_vectors"] for f in features])

        return batch
