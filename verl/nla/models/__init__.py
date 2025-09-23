"""Model wrappers for NLA injection."""

from .nla_wrapper import NLAModelWrapper, InjectionConfig
from .nla_critic_model import AutoModelForCausalLMWithVectorValueHead, NLAVectorValueHead

__all__ = [
    "NLAModelWrapper",
    "InjectionConfig",
    "AutoModelForCausalLMWithVectorValueHead",
    "NLAVectorValueHead",
]