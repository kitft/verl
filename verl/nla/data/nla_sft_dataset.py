"""NLA SFT Dataset with activation vector support."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from transformers import PreTrainedTokenizer

from verl.nla.utils.injection_manager import InjectionTokenManager
from verl.utils.dataset.sft_dataset import SFTDataset


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
        parquet_files: str | ListConfig,
        tokenizer: str | PreTrainedTokenizer,
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
        # Get activation_dim from config - will be inferred from dataset if not provided
        self.activation_dim = config.get("activation_dim", None)

        # Get scale and norm transformations from config
        self.scale = config.get("scale", None)
        self.norm = config.get("norm", None)

        # Initialize base dataset first to get tokenizer
        super().__init__(parquet_files, tokenizer, config)

        # Set up injection token management
        injection_token = config.get("injection_token", None)
        self.injection_manager = InjectionTokenManager(self.tokenizer, injection_token)
        self.injection_token = self.injection_manager.character
        self.injection_token_id = self.injection_manager.token_id

        # Print transformation settings
        print(f"NLASFTDataset: Loading activation vectors with transformations:")
        print(f"  - Normalization: {self.norm if self.norm else 'None'}")
        print(f"  - Scale: {self.scale if self.scale else 'None'}")

        # Load activation vectors
        self._load_activation_vectors()

    def _load_activation_vectors(self):
        """Load activation vectors from the dataframe."""
        # Activation vectors should be in the dataframe
        if "activation_vector" not in self.dataframe.columns:
            raise ValueError("Dataset must contain 'activation_vector' column")

        # Infer activation_dim from first vector if not provided
        if self.activation_dim is None:
            first_vec = self.dataframe.iloc[0]["activation_vector"]
            if isinstance(first_vec, (list, tuple)):
                self.activation_dim = len(first_vec)
            elif hasattr(first_vec, "shape"):
                self.activation_dim = first_vec.shape[-1]
            elif hasattr(first_vec, "__len__"):
                self.activation_dim = len(first_vec)
            else:
                raise ValueError("Cannot infer activation_dim from dataset - please provide it in config")
            print(f"NLASFTDataset: Inferred activation_dim={self.activation_dim} from dataset")

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
                raise ValueError(
                    f"Activation vector dimension mismatch: expected {self.activation_dim}, got {vec.shape[-1]}"
                )

            # Apply normalization if specified
            if self.norm == "unit":
                vec_norm = torch.norm(vec)
                if vec_norm > 0:
                    vec = vec / vec_norm

            # Apply scaling if specified
            if self.scale is not None:
                vec = vec * self.scale

            self.activation_vectors.append(vec)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
            combined_sample.update(
                {
                    "response_ids": critic_sample["response_ids"],
                    "response_attention_mask": critic_sample["response_attention_mask"],
                }
            )
            return combined_sample
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_raw_prompt(self, idx: int) -> str:
        """
        Extract raw prompt content from the prompt field.

        The prompt field is an array of message objects. We extract the first
        'user' role message content as the raw prompt.
        """
        prompt_data = self.prompts[idx]

        # Handle numpy array or list
        if hasattr(prompt_data, "tolist"):
            prompt_data = prompt_data.tolist()
        elif isinstance(prompt_data, str):
            # Already a string, return as-is
            return prompt_data

        # Extract user message content
        for msg in prompt_data:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "").strip()

        raise ValueError(f"No user message found in prompt at index {idx}")

    def _prepare_actor_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Prepare sample for actor training with activation injection.

        The prompt should contain the injection token where the activation
        vector should be injected.
        """
        # Extract raw prompt content instead of using pre-formatted prompt_text
        raw_prompt = self._get_raw_prompt(idx)

        # Temporarily replace prompt with raw content for parent processing
        original_prompt = self.prompts[idx]
        self.prompts[idx] = raw_prompt

        try:
            # Get base sample from parent class (will apply chat template correctly)
            sample = super().__getitem__(idx)
        finally:
            # Restore original prompt
            self.prompts[idx] = original_prompt

        # Attach activation vector
        sample["activation_vectors"] = self.activation_vectors[idx]

        if self.injection_token_id not in sample["input_ids"]:
            raise ValueError(
                "Prompt does not contain the expected injection token. "
                "Ensure the dataset prompt_text includes the injection marker."
            )

        # For Megatron backend: extract response tokens from input_ids using loss_mask
        # loss_mask is 0 for prompt tokens and 1 for response tokens (except last token)
        loss_mask = sample["loss_mask"]
        # Find where response starts (first non-zero in loss_mask)
        response_start_idx = (loss_mask != 0).nonzero(as_tuple=False)
        if len(response_start_idx) > 0:
            response_start = response_start_idx[0].item()
            # Find where response ends (last non-zero in loss_mask + 1 for the masked last token)
            response_end_idx = (loss_mask != 0).nonzero(as_tuple=False)[-1].item() + 1
            # Extract response tokens (excluding padding)
            sample["responses"] = sample["input_ids"][response_start:response_end_idx + 1]
        else:
            # If no response tokens (edge case), create empty tensor
            sample["responses"] = torch.tensor([], dtype=sample["input_ids"].dtype)

        # Skip metadata attachment for now to avoid TensorDict batch issues
        # sample = self._attach_metadata(idx, sample)

        return sample

    def _prepare_critic_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Prepare sample for critic training.

        The critic only needs the response text to predict activation vectors.
        Note: We do NOT add eos_token to match RL behavior where special tokens are masked out.
        """
        response = self.responses[idx]

        # Tokenize response only (no eos_token to match RL behavior)
        # response_with_eos = response + self.tokenizer.eos_token
        response_output = self.tokenizer(
            response,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )

        sample = {
            "input_ids": response_output["input_ids"][0],
            "attention_mask": response_output["attention_mask"][0],
            "activation_vectors": self.activation_vectors[idx],  # Target for critic
        }

        return sample

    def _attach_metadata(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]

        metadata_keys = [
            "sample_uuid",
            "tokenized_source",
            "forward_pass_text",
            "prompt_text",
            "activation_layer",
            "activation_token",
            "activation_token_id",
            "source_message_token_index",
            "source_sequence_token_index",
            "prompt_token_index",
            "is_chat",
            "source",
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


class NLASFTCollator:
    """
    Data collator for NLA SFT that properly handles activation vectors.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate batch of samples, handling activation vectors and variable-length responses.

        Simple collator for FSDP SFT training.
        """
        batch = {}

        # Stack standard tensor fields - these are already padded to same length by dataset
        tensor_keys = ["input_ids", "attention_mask", "loss_mask", "position_ids"]
        for key in tensor_keys:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Stack activation vectors - these should all be same size
        if "activation_vectors" in features[0]:
            batch["activation_vectors"] = torch.stack([f["activation_vectors"] for f in features])

        # Handle responses field (for Megatron backend) with padding if present
        # Note: FSDP doesn't use this field, but we pad it anyway for compatibility
        if "responses" in features[0]:
            # Find max response length in batch
            max_resp_len = max(f["responses"].size(0) for f in features)
            # Pad all responses to max length
            padded_responses = []
            for f in features:
                resp = f["responses"]
                pad_len = max_resp_len - resp.size(0)
                if pad_len > 0:
                    # Pad with pad_token_id
                    padded_resp = torch.cat([resp, torch.full((pad_len,), self.pad_token_id, dtype=resp.dtype)])
                else:
                    padded_resp = resp
                padded_responses.append(padded_resp)
            batch["responses"] = torch.stack(padded_responses)

        return batch


class NLASFTCollatorWithRLFields:
    """
    Data collator for NLA SFT that adds RL-compatible fields for update_policy training.

    This collator adds dummy RL fields to enable SFT training via the RL update_policy path:
    - old_log_probs = 0: Makes the policy ratio exp(new_log_prob - 0) = exp(new_log_prob)
    - advantages = 1: Makes the weight 1 * exp(new_log_prob) = exp(new_log_prob)
    - response_mask: Copy of loss_mask to indicate which tokens are part of the response

    With no clipping, this becomes standard maximum likelihood estimation (SFT).
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate batch of samples with RL-compatible fields for update_policy.
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

        # Handle responses field (for Megatron backend) with padding
        # IMPORTANT: Do this BEFORE creating old_log_probs/advantages which depend on responses shape
        if "responses" in features[0]:
            # Find max response length in batch
            max_resp_len = max(f["responses"].size(0) for f in features)
            # Pad all responses to max length
            padded_responses = []
            for f in features:
                resp = f["responses"]
                pad_len = max_resp_len - resp.size(0)
                if pad_len > 0:
                    # Pad with pad_token_id
                    padded_resp = torch.cat([resp, torch.full((pad_len,), self.pad_token_id, dtype=resp.dtype)])
                else:
                    padded_resp = resp
                padded_responses.append(padded_resp)
            batch["responses"] = torch.stack(padded_responses)

        # For SFT via update_policy: add dummy fields to make it equivalent to supervised learning
        # - response_mask: Extract from loss_mask (response portion only, masks padding)
        # - old_log_probs = 0: ratio becomes exp(new_log_prob - 0) = exp(new_log_prob)
        # - advantages = 1: weight becomes 1 * exp(new_log_prob) = exp(new_log_prob)
        # - With no clipping, this is standard maximum likelihood (SFT)
        #
        # IMPORTANT: Shape these based on responses (response tokens only), not full input_ids,
        # because the actor only computes log_probs for response tokens.
        if "responses" in batch and "loss_mask" in batch:
            batch_size, response_len = batch["responses"].shape
            full_seq_len = batch["loss_mask"].shape[1]

            # Extract response_mask from loss_mask
            # Prompts are left-padded, responses are on the right
            # loss_mask is 1 for response tokens, 0 for prompt/padding
            # Truncate loss_mask from the left to get response portion only
            prompt_len = full_seq_len - response_len
            batch["response_mask"] = batch["loss_mask"][:, prompt_len:]

            batch["old_log_probs"] = torch.zeros((batch_size, response_len), dtype=torch.float32)
            batch["advantages"] = torch.ones((batch_size, response_len), dtype=torch.float32)

        # Stack activation vectors
        if "activation_vectors" in features[0]:
            # Ensure all activation vectors have the same shape
            batch["activation_vectors"] = torch.stack([f["activation_vectors"] for f in features])

        return batch
