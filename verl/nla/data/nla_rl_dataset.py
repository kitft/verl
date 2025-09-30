"""NLA dataset for RL training with activation vectors."""

import torch
import numpy as np
from typing import Dict, Optional, Any, Union, List
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.nla.utils.injection_manager import InjectionTokenManager


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
        data_files: Union[str, List[str]],  # Changed from parquet_files to match parent
        tokenizer: Any,
        config: Optional[Dict] = None,  # Config dict for parent class
        prompt_key: str = "prompt",
        activation_vector_key: str = "activation_vector",
        max_prompt_length: int = 512,
        injection_token: str = None,  # Will be auto-determined from tokenizer
        injection_position: str = "end",  # "start", "end", or "manual"
        activation_dim: int = 768,
        **kwargs
    ):
        """
        Initialize NLA RL dataset.

        Args:
            data_files: Path(s) to parquet files containing prompts and activation vectors
            tokenizer: Tokenizer instance
            config: Configuration dict for base RLHFDataset
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
        self.injection_position = injection_position
        self.activation_dim = activation_dim

        # Create config if not provided
        if config is None:
            config = {
                "prompt_key": prompt_key,
                "max_prompt_length": max_prompt_length,
            }
        else:
            # Update config with our parameters
            config["prompt_key"] = prompt_key
            config["max_prompt_length"] = max_prompt_length

        # Set up injection token management
        self.injection_manager = InjectionTokenManager(tokenizer, injection_token)
        self.injection_token = self.injection_manager.character
        self.injection_token_id = self.injection_manager.token_id

        # Initialize base dataset with correct parameters
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            **kwargs
        )


    def maybe_filter_out_long_prompts(self, dataframe):
        """
        Override to handle message format during filtering.
        """
        if self.max_prompt_length is None:
            return dataframe

        print(f"dataset len: {len(dataframe)}")

        def doc2len(doc) -> int:
            import numpy as np
            # The prompts should already be in message format from our dataset
            messages = doc[self.prompt_key]
            # Convert numpy array to list if needed (parquet storage quirk)
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            return len(
                self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, **self.apply_chat_template_kwargs
                )
            )

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def _build_messages(self, example: dict):
        """
        Override - prompts are already in message format from our dataset.
        """
        messages = example.pop(self.prompt_key)
        # Convert numpy array to list if needed (parquet storage quirk)
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        return messages

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
        activation_vector = sample.pop(self.activation_vector_key, None)

        if activation_vector is None:
            dataframe = getattr(self, "dataframe", None)
            column_names = getattr(dataframe, "column_names", []) if dataframe is not None else []
            if self.activation_vector_key in column_names:
                activation_vector = dataframe[idx][self.activation_vector_key]

        if activation_vector is None:
            raise KeyError(
                "Activation vector missing from dataset row. Ensure the parquet file contains an"
                f" '{self.activation_vector_key}' column with vectors for each example."
            )

        # Convert to tensor if needed
        if isinstance(activation_vector, (list, np.ndarray)):
            activation_vector = torch.tensor(activation_vector, dtype=torch.float32)
        elif not isinstance(activation_vector, torch.Tensor):
            activation_vector = torch.tensor(activation_vector, dtype=torch.float32)

        if activation_vector.dim() == 0:
            activation_vector = activation_vector.unsqueeze(0)

        if activation_vector.shape[-1] != self.activation_dim:
            raise ValueError(
                "Activation vector dimension mismatch: "
                f"expected {self.activation_dim}, got {activation_vector.shape[-1]}"
            )

        sample["activation_vectors"] = activation_vector

        # Add injection token to input_ids if not already present
        if self.injection_token_id not in sample["input_ids"]:
            sample = self._add_injection_token(sample)

        # Store injection token ID for reference
        sample["injection_token_id"] = self.injection_token_id

        response_text = sample.pop("response", None)

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
            "formatted_source",
        ]

        extra_info = sample.get("extra_info", {}) or {}
        for key in metadata_keys:
            value = sample.pop(key, None)
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                extra_info[key] = value

        if response_text is not None:
            extra_info["response"] = response_text

        sample["extra_info"] = extra_info

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

    # Create base config for RLHFDataset
    base_config = {
        "prompt_key": config.get("prompt_key", "prompt"),
        "max_prompt_length": config.get("max_prompt_length", 512),
        "return_raw_chat": config.get("return_raw_chat", True),
    }

    dataset = NLARLDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=base_config,
        prompt_key=config.get("prompt_key", "prompt"),
        activation_vector_key=config.get("activation_vector_key", "activation_vector"),
        max_prompt_length=config.get("max_prompt_length", 512),
        injection_token=config.get("injection_token", None),
        injection_position=config.get("injection_position", "end"),
        activation_dim=config.get("activation_dim", 768),
    )

    return dataset
