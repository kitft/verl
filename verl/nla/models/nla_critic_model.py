"""NLA Critic Model using base model + vector value head."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

@dataclass
class CausalLMOutputWithValue(ModelOutput):
    """Output type for models that output both logits and value predictions."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    value: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class NLAVectorValueHead(nn.Module):
    """
    Value head that outputs a vector instead of a scalar.
    This replaces the standard scalar value head in VERL critics.
    The output is the same dimension as the hidden states (residual stream).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        # No projection needed - we output hidden_size dimensional vectors
        # Just a linear layer for learning but same input/output dim
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) or (batch, hidden_size)
        Returns:
            values: (batch, seq_len, hidden_size) or (batch, hidden_size)
        """
        hidden_states = self.dropout(hidden_states)
        values = self.linear(hidden_states)
        return values


class AutoModelForCausalLMWithVectorValueHead(nn.Module):
    """
    Critic model that uses a copy of the base language model with a vector value head.
    Compatible with VERL's critic infrastructure.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        dropout: float = 0.1,
        **model_kwargs
    ):
        super().__init__()

        # Load the base model (same as actor)
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            **model_kwargs
        )

        # Get hidden size from model config
        self.config = self.pretrained_model.config
        self.hidden_size = self.config.hidden_size

        # Add the vector value head that outputs hidden_size dimensional vectors
        self.v_head = NLAVectorValueHead(
            hidden_size=self.hidden_size,
            dropout=dropout
        )

        # Initialize value head with small random weights
        nn.init.normal_(self.v_head.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.v_head.linear.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """
        Forward pass compatible with VERL's critic interface.

        Returns:
            CausalLMOutputWithValue with:
            - loss: None (not used in critic)
            - logits: LM logits (not used in critic)
            - value: Vector values from the value head (batch, seq_len, activation_dim)
            - hidden_states: Hidden states from the model
        """

        # Get outputs from the base model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need hidden states for the value head
            return_dict=True,
            **kwargs
        )

        # Get the last hidden states from the final layer
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Extract last token for each sequence (where we'll get the activation vector)
        if attention_mask is not None:
            # Find the last valid token position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_size = hidden_states.shape[0]
            last_hidden = hidden_states[torch.arange(batch_size), seq_lengths]  # (batch, hidden_size)
        else:
            # If no mask, just take the last position
            last_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)

        # Apply value head to get the activation vector (hidden_size dimensional)
        activation_vector = self.v_head(last_hidden)  # (batch, hidden_size)

        # For compatibility, expand back to sequence dimension with zeros except last position
        batch_size, seq_len = hidden_states.shape[:2]
        values = torch.zeros(batch_size, seq_len, self.hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        if attention_mask is not None:
            values[torch.arange(batch_size), seq_lengths] = activation_vector
        else:
            values[:, -1] = activation_vector

        # Return in format expected by VERL
        # The dp_critic.py expects output[2] to be the values (line 108, 132)
        if not return_dict:
            return (outputs.loss, outputs.logits, values) + outputs.hidden_states

        return CausalLMOutputWithValue(
            loss=outputs.loss,
            logits=outputs.logits,
            value=values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a pretrained model with vector value head.
        The activation dimension is automatically set to the model's hidden size.

        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional arguments for model loading
        """
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs
        )