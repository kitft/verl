"""Autoencoder-style critic that outputs activation vectors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class AutoencoderCriticOutput:
    """Output from the autoencoder critic."""

    predicted_activation: torch.Tensor  # (batch_size, activation_dim)
    hidden_states: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class NLAAutoencoderCritic(nn.Module):
    """
    Critic that takes generated responses and outputs activation vectors.
    Used for autoencoder-style training with MSE loss against original activations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        activation_dim: int = 768,
        hidden_dim: Optional[int] = None,
        use_pooling: str = "last",  # "last", "mean", "max", "cls", or "weighted"
        dropout: float = 0.1,
        num_projection_layers: int = 2,
        cls_token_id: Optional[int] = None,  # Token ID for CLS token if using cls pooling
        max_seq_length: int = 2048,  # Maximum sequence length for weighted pooling
    ):
        super().__init__()
        self.base_model = base_model
        self.activation_dim = activation_dim
        self.hidden_dim = hidden_dim or self._infer_hidden_dim()
        self.use_pooling = use_pooling
        self.cls_token_id = cls_token_id
        self.max_seq_length = max_seq_length

        # Build projection head from hidden states to activation vectors
        self.projection_head = self._build_projection_head(
            input_dim=self.hidden_dim,
            output_dim=activation_dim,
            num_layers=num_projection_layers,
            dropout=dropout,
        )

        # Optional: learnable pooling weights
        if use_pooling == "weighted":
            # Initialize with small random values for learnable position-wise attention
            self.pooling_weights = nn.Parameter(torch.randn(max_seq_length) * 0.01)

        # Warn if using cls pooling without cls_token_id
        if use_pooling == "cls" and cls_token_id is None:
            import warnings
            warnings.warn(
                "Using 'cls' pooling without providing cls_token_id. "
                "Assuming CLS token is already appended to input sequences."
            )

    def _infer_hidden_dim(self) -> int:
        """Infer hidden dimension from base model."""
        for attr in ["hidden_size", "d_model", "embed_dim", "hidden_dim"]:
            if hasattr(self.base_model.config, attr):
                return getattr(self.base_model.config, attr)
        raise ValueError("Could not infer hidden dimension from base model")

    def _build_projection_head(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Build MLP projection head."""
        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)

        layers = []
        hidden_dim = (input_dim + output_dim) // 2

        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence hidden states into a single vector."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.use_pooling == "last":
            # Use last non-padding token
            if attention_mask is not None:
                # Find last real token position for each batch
                seq_lengths = (attention_mask.sum(dim=1) - 1).long()
                batch_idx = torch.arange(batch_size, device=hidden_states.device)
                pooled = hidden_states[batch_idx, seq_lengths]
            else:
                pooled = hidden_states[:, -1]

        elif self.use_pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.use_pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states = hidden_states.masked_fill(~mask.bool(), -1e9)
            pooled = hidden_states.max(dim=1)[0]

        elif self.use_pooling == "cls":
            # Assumes a CLS token has been appended at the end for autoregressive models
            # This gives the CLS token full context of the sequence
            if attention_mask is not None:
                # Find last real token position (the appended CLS token)
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(batch_size, device=hidden_states.device)
                pooled = hidden_states[batch_idx, seq_lengths]
            else:
                pooled = hidden_states[:, -1]

        elif self.use_pooling == "weighted":
            # Learnable weighted average with proper position-wise weights
            # Slice weights to current sequence length
            weights = torch.softmax(self.pooling_weights[:seq_len], dim=0)

            # Expand weights for batch processing
            weights = weights.unsqueeze(0).expand(batch_size, -1)

            if attention_mask is not None:
                # Apply attention mask
                weights = weights * attention_mask.float()
                # Renormalize after masking
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

            # Apply weights to hidden states
            pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.use_pooling}")

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        append_cls_token: bool = None,  # Auto-append CLS token if using cls pooling
        **kwargs,
    ) -> AutoencoderCriticOutput:
        """
        Forward pass: encode response text to predicted activation vector.

        Args:
            input_ids: Generated response token IDs (batch_size, seq_len)
            attention_mask: Attention mask for responses
            return_hidden_states: Whether to return intermediate hidden states
            append_cls_token: Whether to append CLS token (defaults to True if using cls pooling)

        Returns:
            AutoencoderCriticOutput with predicted activation vectors
        """
        # Auto-append CLS token if using cls pooling and token ID is provided
        if append_cls_token is None:
            append_cls_token = self.use_pooling == "cls" and self.cls_token_id is not None

        if append_cls_token and self.cls_token_id is not None:
            batch_size = input_ids.shape[0]
            cls_tokens = torch.full((batch_size, 1), self.cls_token_id,
                                   dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, cls_tokens], dim=1)

            if attention_mask is not None:
                cls_mask = torch.ones((batch_size, 1),
                                     dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, cls_mask], dim=1)

        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Extract last layer hidden states
        if hasattr(outputs, "hidden_states"):
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs.last_hidden_state

        # Pool hidden states to get sequence representation
        pooled_output = self.pool_hidden_states(last_hidden_state, attention_mask)

        # Project to activation space
        predicted_activation = self.projection_head(pooled_output)

        return AutoencoderCriticOutput(
            predicted_activation=predicted_activation,
            hidden_states=outputs.hidden_states if return_hidden_states else None,
            last_hidden_state=last_hidden_state,
        )

    def compute_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target activations."""
        return F.mse_loss(predicted, target, reduction=reduction)

    def compute_rewards(
        self,
        predicted_activations: torch.Tensor,
        target_activations: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute rewards based on MSE between predicted and target activations.
        Lower MSE = higher reward.

        Args:
            predicted_activations: Predicted activation vectors from critic
            target_activations: Original activation vectors
            normalize: Whether to normalize rewards

        Returns:
            Rewards tensor (batch_size,)
        """
        # Compute per-example MSE
        mse = ((predicted_activations - target_activations) ** 2).mean(dim=-1)

        # Convert MSE to reward (lower MSE = higher reward)
        # Use negative MSE or transform it
        rewards = -mse  # Simple negative MSE

        # Alternative: exponential transformation
        # rewards = torch.exp(-mse)

        # Alternative: bounded transformation
        # rewards = 1.0 / (1.0 + mse)

        if normalize:
            # Normalize rewards to have zero mean and unit variance
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return rewards


class NLAValueHead(nn.Module):
    """
    Simplified value head for the critic that outputs vector predictions.
    Can be used as a drop-in replacement for standard value heads.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation_dim: int,
        intermediate_dim: Optional[int] = None,
    ):
        super().__init__()
        intermediate_dim = intermediate_dim or hidden_dim // 2

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, activation_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass: hidden states -> activation vector."""
        return self.layers(hidden_states)