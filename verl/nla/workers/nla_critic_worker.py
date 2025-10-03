"""NLA Critic Worker with vector value head support."""

import torch

from verl.workers.fsdp_workers import CriticWorker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu


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
        """Predict activation vectors for the final response token."""
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
                with torch.no_grad():
                    outputs = self.critic_module(
                        input_ids=response_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                    if attention_mask is not None:
                        seq_lengths = attention_mask.sum(dim=1) - 1
                        seq_lengths = seq_lengths.clamp(min=0)
                        indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                        activations = hidden_states[indices, seq_lengths]
                    else:
                        activations = hidden_states[:, -1, :]
            finally:
                if was_training:
                    self.critic_module.train(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        return activations
