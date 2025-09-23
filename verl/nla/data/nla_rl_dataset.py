"""NLA dataset for RL training with activation vectors."""

import torch
import numpy as np
from typing import Dict, Optional, Any, Union, List
from verl.utils.dataset.rl_dataset import RLHFDataset


class NLARLDataset(RLHFDataset):
    """
    RL dataset for NLA that includes activation vectors and injection tokens.

    This dataset extends the base RLDataset to:
    1. Load activation vectors from the parquet data
    2. Add injection tokens to prompts
    3. Package data in a format compatible with NLA trainers
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: Any,
        prompt_key: str = "prompt",
        activation_vector_key: str = "activation_vector",
        max_prompt_length: int = 512,
        injection_token: str = "<INJECT>",
        injection_position: str = "end",  # "start", "end", or "manual"
        activation_dim: int = 768,
        **kwargs
    ):
        """
        Initialize NLA RL dataset.

        Args:
            parquet_files: Path(s) to parquet files containing prompts and activation vectors
            tokenizer: Tokenizer instance
            prompt_key: Column name for prompts in the parquet file
            activation_vector_key: Column name for activation vectors
            max_prompt_length: Maximum length for prompts
            injection_token: Special token to mark injection position
            injection_position: Where to inject the activation token
            activation_dim: Dimension of activation vectors
            **kwargs: Additional arguments for base RLDataset
        """
        # Store NLA-specific parameters
        self.activation_vector_key = activation_vector_key
        self.injection_token = injection_token
        self.injection_position = injection_position
        self.activation_dim = activation_dim

        # Ensure injection token is in tokenizer
        self.injection_token_id = self._ensure_injection_token(tokenizer)

        # Initialize base dataset
        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            prompt_key=prompt_key,
            max_prompt_length=max_prompt_length,
            **kwargs
        )

    def _ensure_injection_token(self, tokenizer) -> int:
        """Ensure the injection token exists in the tokenizer."""
        if self.injection_token not in tokenizer.get_vocab():
            # Add special token to tokenizer
            special_tokens = {"additional_special_tokens": [self.injection_token]}
            tokenizer.add_special_tokens(special_tokens)
            print(f"Added injection token '{self.injection_token}' to tokenizer")

        injection_token_id = tokenizer.convert_tokens_to_ids(self.injection_token)
        return injection_token_id

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample with activation vector.

        Returns:
            Dict containing:
            - All standard RL dataset fields (input_ids, attention_mask, etc.)
            - activation_vectors: The activation vector for this sample
            - injection_token_id: The token ID for injection marker
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)

        # Load activation vector for this sample
        if self.activation_vector_key in self.data.columns:
            activation_vector = self.data.iloc[idx][self.activation_vector_key]

            # Convert to tensor if needed
            if isinstance(activation_vector, (list, np.ndarray)):
                activation_vector = torch.tensor(activation_vector, dtype=torch.float32)
            elif not isinstance(activation_vector, torch.Tensor):
                activation_vector = torch.tensor(activation_vector, dtype=torch.float32)

            # Ensure correct shape
            if activation_vector.dim() == 0:
                activation_vector = activation_vector.unsqueeze(0)

            # Validate dimension
            if activation_vector.shape[-1] != self.activation_dim:
                raise ValueError(
                    f"Activation vector dimension mismatch: "
                    f"expected {self.activation_dim}, got {activation_vector.shape[-1]}"
                )

            sample["activation_vectors"] = activation_vector
        else:
            # Create a zero vector if no activation provided (for testing)
            sample["activation_vectors"] = torch.zeros(self.activation_dim, dtype=torch.float32)

        # Add injection token to input_ids if not already present
        if self.injection_token_id not in sample["input_ids"]:
            sample = self._add_injection_token(sample)

        # Store injection token ID for reference
        sample["injection_token_id"] = self.injection_token_id

        return sample

    def _add_injection_token(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add injection token to the prompt.

        Args:
            sample: Dictionary containing input_ids and attention_mask

        Returns:
            Updated sample with injection token added
        """
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]

        # Find the position to insert the injection token
        if self.injection_position == "manual":
            # Assume it's already in the text (user added it manually)
            return sample

        # Find the end of the actual prompt (before padding)
        actual_length = attention_mask.sum().item()

        if self.injection_position == "end":
            # Insert at the end of the prompt
            insert_position = actual_length
        elif self.injection_position == "start":
            # Insert at the beginning (after any system tokens)
            insert_position = 1  # Assuming position 0 might be a special token
        else:
            raise ValueError(f"Unknown injection_position: {self.injection_position}")

        # Create new tensors with injection token
        new_input_ids = torch.zeros_like(input_ids)
        new_attention_mask = torch.zeros_like(attention_mask)

        # Copy tokens before injection point
        if insert_position > 0:
            new_input_ids[:insert_position] = input_ids[:insert_position]
            new_attention_mask[:insert_position] = attention_mask[:insert_position]

        # Insert injection token
        if insert_position < len(input_ids):
            new_input_ids[insert_position] = self.injection_token_id
            new_attention_mask[insert_position] = 1

            # Copy remaining tokens (shifted by 1)
            remaining_length = min(actual_length - insert_position, len(input_ids) - insert_position - 1)
            if remaining_length > 0:
                new_input_ids[insert_position + 1:insert_position + 1 + remaining_length] = \
                    input_ids[insert_position:insert_position + remaining_length]
                new_attention_mask[insert_position + 1:insert_position + 1 + remaining_length] = \
                    attention_mask[insert_position:insert_position + remaining_length]

        sample["input_ids"] = new_input_ids
        sample["attention_mask"] = new_attention_mask

        return sample


def create_nla_rl_dataset(
    data_files: Union[str, List[str]],
    tokenizer: Any,
    config: Optional[Dict] = None,
) -> NLARLDataset:
    """
    Factory function to create NLA RL dataset.

    Args:
        data_files: Path(s) to data files
        tokenizer: Tokenizer instance
        config: Configuration dictionary

    Returns:
        NLARLDataset instance
    """
    config = config or {}

    dataset = NLARLDataset(
        parquet_files=data_files,
        tokenizer=tokenizer,
        prompt_key=config.get("prompt_key", "prompt"),
        activation_vector_key=config.get("activation_vector_key", "activation_vector"),
        max_prompt_length=config.get("max_prompt_length", 512),
        injection_token=config.get("injection_token", "<INJECT>"),
        injection_position=config.get("injection_position", "end"),
        activation_dim=config.get("activation_dim", 768),
    )

    return dataset