"""NLA Critic Model using base model + vector value head."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    """Output type for models that output both logits and value predictions."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    value: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class NLAVectorValueHead(nn.Module):
    """
    Value head that outputs a vector instead of a scalar.
    This replaces the standard scalar value head in VERL critics.
    The output is the same dimension as the hidden states (residual stream).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        # No projection needed - we output hidden_size dimensional vectors
        # Just a linear layer for learning but same input/output dim
        # self.linear = nn.Linear(hidden_size, hidden_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) or (batch, hidden_size)
        Returns:
            values: (batch, seq_len, hidden_size) or (batch, hidden_size)
        """
        # hidden_states = self.dropout(hidden_states)
        # values = self.linear(hidden_states)
        return hidden_states


class AutoModelForCausalLMWithVectorValueHead(nn.Module):
    """
    Critic model that uses a copy of the base language model with a vector value head.
    Compatible with VERL's critic infrastructure.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = None,
        *,
        pretrained_model: nn.Module = None,
        dropout: float = 0.1,
        output_layer_index: int = 20,
        **model_kwargs,
    ):
        super().__init__()

        if pretrained_model is not None:
            # Use pre-loaded model (allows truncation before wrapping)
            self.pretrained_model = pretrained_model
        elif pretrained_model_name_or_path is not None:
            # Load the base model (same as actor)
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
        else:
            raise ValueError("Either pretrained_model_name_or_path or pretrained_model must be provided.")

        # Get hidden size from model config
        self.config = self.pretrained_model.config
        self.hidden_size = self.config.hidden_size
        self.output_layer_index = output_layer_index

        # No value head - we directly use the hidden states as activation vectors
        # This simplifies FSDP wrapping since we don't have extra modules
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithValue:
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
            **kwargs,
        )

        # Get the last hidden states from the final layer

        # hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # # Extract last token for each sequence (where we'll get the activation vector)
        # if attention_mask is not None:
        #     # Find the last valid token position for each sequence
        #     seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        #     batch_size = hidden_states.shape[0]
        #     last_hidden = hidden_states[torch.arange(batch_size), seq_lengths]  # (batch, hidden_size)
        # else:
        #     raise ValueError("Attention mask is required for last pooling")
        #     # If no mask, just take the last position
        #     last_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)

        # if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        #    last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[self.output_layer_index]
        else:
            raise ValueError(
                f"Last hidden state or hidden states are not available in the critic output: available keys are {outputs.keys()} and which are none? {[key is None for key in outputs.keys()]}"
            )
        # elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        #     raise ValueError("Last hidden state is not available in the critic output. this has been normed")
        # else:
        #     print(f"type of model = {self.pretrained_model}")
        #     print(f"type of model = {type(self.pretrained_model)}")
        #     print(f"attrs of model = {dir(self.pretrained_model)}")
        #     import inspect

        #     print(f"args and kwargs accepted by __call__ = {inspect.signature(self.pretrained_model.__call__)}")
        #     raise ValueError(
        #         f"Last hidden state or hidden states are not available in the critic output: available keys are {outputs.keys()} and which are none? {[key is None for key in outputs.keys()]}"
        #     )
        indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        last_hidden = last_hidden_state[indices, seq_lengths]  # (batch, hidden_size) (batch, hidden_size)
        # print(
        #     f"extracting layer -1 of available {len(outputs.hidden_states)} hidden states, each of shape {outputs.hidden_states[0].shape}"
        # )  #
        # print(f"extracting at indices {indices} and seq_lengths {seq_lengths}")

        # Directly use the last hidden state as activation vector (with optional dropout)
        # activation_vector = self.dropout(last_hidden)  # (batch, hidden_size)
        # no dropout
        activation_vector = last_hidden
        # Return activation vector as (batch, hidden_size)
        values = activation_vector
        # print("shape of values inside the forward pass:", values.shape)
        # raise ValueError("Stop here - not properly implemented")

        if not return_dict:
            return (None, None, values)

        return CausalLMOutputWithValue(
            loss=None,
            logits=None,
            value=values,
            hidden_states=outputs.hidden_states,  # outputs.hidden_states,  # Return full tuple from base model
            attentions=None,
        )

    def can_generate(self) -> bool:
        """Returns whether this model can generate.

        The critic model is not designed for generation, only for
        extracting activation vectors.
        """
        return False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a pretrained model with vector value head.
        The activation dimension is automatically set to the model's hidden size.

        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional arguments for model loading
        """
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
