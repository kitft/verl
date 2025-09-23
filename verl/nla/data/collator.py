"""Data collator for batching NLA data."""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class NLADataCollator:
    """Collates batches of NLA data."""

    pad_token_id: int = 0
    padding_side: str = "right"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate list of samples into a batch."""
        batch = {}

        # Stack input_ids and attention_mask
        if "input_ids" in features[0]:
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])

        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])

        # Stack activation vectors
        if "activation_vector" in features[0]:
            # Handle variable-size activation vectors by padding if needed
            activations = [f["activation_vector"] for f in features]
            max_dim = max(a.shape[-1] for a in activations)

            padded_activations = []
            for a in activations:
                if a.shape[-1] < max_dim:
                    # Pad with zeros
                    padding = torch.zeros(
                        *a.shape[:-1],
                        max_dim - a.shape[-1],
                        dtype=a.dtype
                    )
                    a = torch.cat([a, padding], dim=-1)
                padded_activations.append(a)

            batch["activation_vectors"] = torch.stack(padded_activations)

        # Collect injection positions (list of lists)
        if "injection_positions" in features[0]:
            batch["injection_positions"] = [f["injection_positions"] for f in features]

        # Handle RL-specific fields
        if "response_ids" in features[0]:
            batch["response_ids"] = torch.stack([f["response_ids"] for f in features])

        if "response_mask" in features[0]:
            batch["response_mask"] = torch.stack([f["response_mask"] for f in features])

        if "reward" in features[0]:
            batch["rewards"] = torch.stack([f["reward"] for f in features])

        # Handle labels if present
        if "labels" in features[0]:
            if isinstance(features[0]["labels"], torch.Tensor):
                batch["labels"] = torch.stack([f["labels"] for f in features])
            else:
                batch["labels"] = torch.tensor([f["labels"] for f in features])

        return batch


class NLADynamicCollator(NLADataCollator):
    """Dynamic collator that handles variable-length sequences."""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with dynamic padding to max length in batch."""
        batch = {}

        # Find max lengths in batch
        if "input_ids" in features[0]:
            max_length = max(
                len(f["input_ids"]) if not isinstance(f["input_ids"], torch.Tensor)
                else f["input_ids"].shape[0]
                for f in features
            )

            # Pad sequences to max length
            input_ids_list = []
            attention_mask_list = []

            for f in features:
                ids = f["input_ids"]
                mask = f.get("attention_mask", torch.ones_like(ids))

                # Convert to tensor if needed
                if not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids)
                    mask = torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask

                # Pad if needed
                curr_len = ids.shape[0]
                if curr_len < max_length:
                    padding_len = max_length - curr_len
                    if self.padding_side == "right":
                        ids = torch.cat([
                            ids,
                            torch.full((padding_len,), self.pad_token_id, dtype=ids.dtype)
                        ])
                        mask = torch.cat([
                            mask,
                            torch.zeros(padding_len, dtype=mask.dtype)
                        ])
                    else:  # left padding
                        ids = torch.cat([
                            torch.full((padding_len,), self.pad_token_id, dtype=ids.dtype),
                            ids
                        ])
                        mask = torch.cat([
                            torch.zeros(padding_len, dtype=mask.dtype),
                            mask
                        ])

                input_ids_list.append(ids)
                attention_mask_list.append(mask)

            batch["input_ids"] = torch.stack(input_ids_list)
            batch["attention_mask"] = torch.stack(attention_mask_list)

        # Handle activation vectors
        if "activation_vector" in features[0]:
            activations = [f["activation_vector"] for f in features]
            # Ensure all tensors
            activations = [
                torch.tensor(a) if not isinstance(a, torch.Tensor) else a
                for a in activations
            ]
            batch["activation_vectors"] = torch.stack(activations)

        # Handle other fields using parent method
        remaining_batch = super().__call__(features)
        batch.update({
            k: v for k, v in remaining_batch.items()
            if k not in batch
        })

        return batch