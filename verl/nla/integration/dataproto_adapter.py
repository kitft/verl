"""Adapter for converting between NLA data format and verl DataProto."""

import torch
from typing import Dict, List, Optional, Any
from verl.protocol import DataProto


class NLADataProtoAdapter:
    """Converts between NLA data format and verl's DataProto."""

    def __init__(self, injection_token: str = "<INJECT>"):
        self.injection_token = injection_token

    def add_activation_vectors_to_dataproto(
        self,
        data_proto: DataProto,
        activation_vectors: torch.Tensor,
        injection_positions: Optional[List[List[int]]] = None,
    ) -> DataProto:
        """Add activation vectors to a DataProto object."""
        meta_info = data_proto.meta_info if data_proto.meta_info is not None else {}
        batch = data_proto.batch

        # Ensure we do not mutate other references in-place
        if meta_info is data_proto.meta_info:
            meta_info = dict(meta_info)

        meta_info["has_nla"] = True
        if injection_positions is not None:
            meta_info["injection_positions"] = injection_positions

        if batch is None:
            raise ValueError("DataProto batch must exist to attach activation vectors")
        batch["activation_vectors"] = activation_vectors
        data_proto.meta_info = meta_info

        return data_proto

    def extract_activation_vectors_from_dataproto(
        self, data_proto: DataProto
    ) -> Optional[torch.Tensor]:
        """Extract activation vectors from a DataProto object."""
        batch = getattr(data_proto, "batch", None)
        if batch is not None and "activation_vectors" in batch.keys():
            return batch["activation_vectors"]

        return None

    def extract_injection_positions_from_dataproto(
        self, data_proto: DataProto
    ) -> Optional[List[List[int]]]:
        """Extract injection positions from a DataProto object."""
        meta_info = getattr(data_proto, "meta_info", None)
        if meta_info:
            return meta_info.get("injection_positions")
        return None

    def find_injection_positions_in_prompts(
        self,
        prompts: DataProto,
        tokenizer: Any,
    ) -> List[List[int]]:
        """Find positions of injection tokens in prompts."""
        injection_token_id = tokenizer.convert_tokens_to_ids(self.injection_token)
        positions = []

        input_ids = prompts.batch["input_ids"]
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            batch_positions = []
            for pos, token_id in enumerate(input_ids[batch_idx]):
                if token_id == injection_token_id:
                    batch_positions.append(pos)
            positions.append(batch_positions)

        return positions

    def prepare_nla_prompts(
        self,
        prompts: List[str],
        activation_vectors: torch.Tensor,
        tokenizer: Any,
        max_length: int = 512,
    ) -> DataProto:
        """Prepare prompts with activation vectors as DataProto."""
        # Tokenize prompts
        encoded = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Create DataProto
        data_proto = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            meta_info={},
        )

        # Find injection positions
        injection_positions = self.find_injection_positions_in_prompts(
            data_proto, tokenizer
        )

        # Add activation vectors
        data_proto = self.add_activation_vectors_to_dataproto(
            data_proto, activation_vectors, injection_positions
        )

        return data_proto

    def merge_nla_with_existing_batch(
        self,
        batch: Dict[str, torch.Tensor],
        activation_vectors: torch.Tensor,
        injection_positions: Optional[List[List[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Merge NLA data with an existing batch dictionary."""
        batch = batch.copy()
        batch["activation_vectors"] = activation_vectors

        if injection_positions is not None:
            batch["injection_positions"] = injection_positions

        return batch
