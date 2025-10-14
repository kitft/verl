"""NLA Critic Worker with vector value head support."""

import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import CriticWorker


class NLACriticWorker(CriticWorker):
    """Critic worker that adapts the standard VERL critic to return activation vectors."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize base critic components then wrap with NLA critic behaviour."""
        super().init_model()

        from verl.nla.workers.nla_critic import NLADataParallelCritic

        self.critic = NLADataParallelCritic(
            config=self.config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            tokenizer=self.tokenizer,  # Pass tokenizer for critic prompt support
        )

    def compute_activation_predictions(self, response_ids, attention_mask):
        """Predict activation vectors for the final response token.

        Delegates to the critic's forward logic to ensure consistency with training,
        including critic prompt prepending if configured.
        """
        from verl import DataProto

        device = get_device_id()
        response_ids = response_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        with self.ulysses_sharding_manager:
            was_training = self.critic_module.training
            try:
                self.critic_module.eval()

                # Create minimal DataProto for forward pass
                batch_dict = {
                    "responses": response_ids,
                    "input_ids": response_ids,
                    "attention_mask": attention_mask,
                    "position_ids": torch.zeros_like(attention_mask, dtype=torch.long),
                }
                data = DataProto(batch=batch_dict)

                # Use critic's forward logic (handles prompt prepending, etc.)
                with torch.no_grad():
                    full_activations = self.critic._compute_values_internal(data, enable_grads=False)

                    # Extract last token using critic's pooling logic
                    response_mask = attention_mask
                    activations = self.critic.extract_predicted_activations(
                        full_activations, response_mask, pooling="last"
                    )
            finally:
                if was_training:
                    self.critic_module.train(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        return activations
