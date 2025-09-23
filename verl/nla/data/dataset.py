"""Dataset classes for handling activation vectors alongside text data."""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class NLABatch:
    """Batch container for NLA data."""

    input_ids: torch.Tensor
    activation_vectors: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    injection_positions: Optional[List[List[int]]] = None
    metadata: Optional[Dict[str, Any]] = None


class NLADataset(Dataset):
    """Dataset for handling prompts with activation vectors."""

    def __init__(
        self,
        prompts: List[str],
        activation_vectors: Union[List[np.ndarray], np.ndarray, torch.Tensor],
        tokenizer: Any,
        injection_token: str = "<INJECT>",
        max_length: int = 512,
        labels: Optional[List[Any]] = None,
    ):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.injection_token = injection_token
        self.max_length = max_length
        self.labels = labels

        # Convert activation vectors to tensors
        if isinstance(activation_vectors, list):
            self.activation_vectors = [
                torch.tensor(av, dtype=torch.float32) if not isinstance(av, torch.Tensor) else av
                for av in activation_vectors
            ]
        elif isinstance(activation_vectors, np.ndarray):
            self.activation_vectors = torch.tensor(activation_vectors, dtype=torch.float32)
        else:
            self.activation_vectors = activation_vectors

        # Add injection token to tokenizer if not present
        self._setup_injection_token()

    def _setup_injection_token(self):
        """Add injection token to tokenizer vocabulary."""
        if self.injection_token not in self.tokenizer.get_vocab():
            # Add as special token
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [self.injection_token]
            })
            self.injection_token_id = self.tokenizer.convert_tokens_to_ids(self.injection_token)
        else:
            self.injection_token_id = self.tokenizer.convert_tokens_to_ids(self.injection_token)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        prompt = self.prompts[idx]

        # Tokenize prompt
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Get activation vector
        if isinstance(self.activation_vectors, list):
            activation_vector = self.activation_vectors[idx]
        else:
            activation_vector = self.activation_vectors[idx]

        # Find injection positions
        input_ids = encoded["input_ids"].squeeze(0)
        injection_positions = (input_ids == self.injection_token_id).nonzero(as_tuple=True)[0].tolist()

        result = {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "activation_vector": activation_vector,
            "injection_positions": injection_positions,
        }

        if self.labels is not None:
            result["labels"] = self.labels[idx]

        return result


class NLARLDataset(NLADataset):
    """Extended dataset for RL training with activation vectors."""

    def __init__(
        self,
        prompts: List[str],
        activation_vectors: Union[List[np.ndarray], np.ndarray, torch.Tensor],
        tokenizer: Any,
        responses: Optional[List[str]] = None,
        rewards: Optional[List[float]] = None,
        injection_token: str = "<INJECT>",
        max_prompt_length: int = 256,
        max_response_length: int = 256,
    ):
        super().__init__(
            prompts=prompts,
            activation_vectors=activation_vectors,
            tokenizer=tokenizer,
            injection_token=injection_token,
            max_length=max_prompt_length,
        )
        self.responses = responses
        self.rewards = rewards
        self.max_response_length = max_response_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base data
        item = super().__getitem__(idx)

        # Add RL-specific fields
        if self.responses is not None:
            response = self.responses[idx]
            response_encoded = self.tokenizer(
                response,
                truncation=True,
                max_length=self.max_response_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["response_ids"] = response_encoded["input_ids"].squeeze(0)
            item["response_mask"] = response_encoded["attention_mask"].squeeze(0)

        if self.rewards is not None:
            item["reward"] = torch.tensor(self.rewards[idx], dtype=torch.float32)

        return item