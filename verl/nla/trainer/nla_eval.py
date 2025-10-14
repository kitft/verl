"""
NLA evaluation module for periodic string-matching evaluations during RL training.

This module provides functionality to evaluate the NLA autoencoder pipeline:
1. Load pre-computed evaluation activations
2. Run actor generation with activation injection
3. Check if expected strings appear in completions
4. Return accuracy metrics

Designed to be called periodically during training to track reconstruction quality.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

from verl.nla.utils.injection_manager import InjectionTokenManager
from verl.protocol import DataProto

logger = logging.getLogger(__name__)


class NLAStringMatchEvaluator:
    """
    Evaluates NLA autoencoder quality using string matching on known prompts.

    Example usage:
        evaluator = NLAStringMatchEvaluator(
            eval_files={"QA": "data/eval/nla_eval_qa.parquet"},
            tokenizer=tokenizer,
        )
        metrics = evaluator.evaluate(actor_rollout_wg, max_new_tokens=50)
    """

    def __init__(
        self,
        eval_files: dict[str, str] | str,
        tokenizer,
        max_samples: Optional[int] = None,
        template: Optional[str] = None,
    ):
        """
        Initialize evaluator with pre-computed activation datasets.

        Args:
            eval_files: Dict mapping dataset names to file paths, or single file path string
            tokenizer: HuggingFace tokenizer
            max_samples: Optional limit on number of eval samples per dataset
            template: Template for evaluation prompts
        """
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.template = template

        # Convert single file to dict format
        if isinstance(eval_files, str):
            eval_files = {"default": eval_files}

        # Convert OmegaConf DictConfig to regular dict for consistency
        try:
            from omegaconf import DictConfig

            if isinstance(eval_files, DictConfig):
                eval_files = dict(eval_files)
        except ImportError:
            pass

        self.datasets = {}
        for dataset_name, eval_file in eval_files.items():
            self.datasets[dataset_name] = self._load_eval_data(eval_file, dataset_name)

        total_samples = sum(len(data) for data in self.datasets.values())
        logger.info(f"Loaded {len(self.datasets)} datasets with {total_samples} total samples")

    def _load_eval_data(self, eval_file: str, dataset_name: str) -> list[dict]:
        """Load evaluation activations from parquet file."""
        eval_path = Path(eval_file)
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Eval file not found: {eval_path}\n"
                f"Generate it using: python scripts/nla_eval/generate_eval_activations.py"
            )

        df = pd.read_parquet(eval_path)

        if self.max_samples is not None:
            df = df.head(self.max_samples)

        eval_data = df.to_dict("records")
        logger.info(f"Loaded {len(eval_data)} samples for dataset '{dataset_name}' from {eval_path}")
        return eval_data

    def prepare_eval_batch(self, eval_data: list[dict]) -> DataProto:
        """
        Prepare DataProto batch for evaluation.

        Args:
            eval_data: List of evaluation records

        Returns:
            DataProto with prompts, activation vectors, and metadata
        """
        prompts = []
        activation_vectors = []
        expected_completions = []
        descriptions = []
        source_prompts = []

        injection_character = InjectionTokenManager(self.tokenizer).character
        prompt_template = self.template.format(injection_character=injection_character)
        for record in eval_data:
            # Use target_chat if available (chat-formatted), otherwise target_prompt
            # prompt_text = record.get("target_chat", record["target_prompt"])
            # pompt_text = record.get("source_prompt", record["target_prompt"])
            source_prompt = record["source_prompt"]

            # Apply chat formatting to the prompt
            messages = [{"role": "user", "content": prompt_template}]
            chat_formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(chat_formatted_prompt)
            activation_vectors.append(record["activation_vector"])
            expected_completions.append(record["expected_completions"])
            descriptions.append(record.get("description", "unknown"))
            source_prompts.append(source_prompt)
        if prompts:
            logger.info(
                f"Prepared batch with {len(prompts)} samples. First prompt: '{prompts[0][:100]}...' "
                f"Expected: '{expected_completions[0]}' Source: '{source_prompts[0][:50]}...'"
            )

        # Tokenize prompts
        # Note: prompts are already chat-formatted if target_chat exists
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Convert activation vectors to tensor (handle both list and numpy inputs)
        import numpy as np

        if isinstance(activation_vectors[0], list):
            activation_tensor = torch.tensor(activation_vectors, dtype=torch.float32)
        elif isinstance(activation_vectors[0], np.ndarray):
            activation_tensor = torch.from_numpy(np.stack(activation_vectors)).float()
        else:
            # Already torch tensors
            activation_tensor = torch.stack(activation_vectors).float()

        # Create position_ids (required by rollout)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        batch_size, seq_len = input_ids.shape

        # Create position_ids based on attention_mask (handle left-padding)
        position_ids = torch.zeros_like(input_ids)
        for i in range(batch_size):
            # Find first non-padded token
            valid_length = attention_mask[i].sum().item()
            start_pos = seq_len - valid_length
            position_ids[i, start_pos:] = torch.arange(valid_length)

        # Create DataProto
        import numpy as np
        from tensordict import TensorDict

        batch = DataProto(
            batch=TensorDict(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "activation_vectors": activation_tensor,
                },
                batch_size=batch_size,
            ),
            non_tensor_batch={
                "prompt_text": np.array(prompts, dtype=object),
                "expected_completions": np.array(expected_completions, dtype=object),
                "descriptions": np.array(descriptions, dtype=object),
                "source_prompt": np.array(source_prompts, dtype=object),
            },
            meta_info={
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            },
        )

        return batch

    def evaluate(
        self,
        actor_rollout_wg,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_generations_per_prompt: int = 1,
        global_steps: int = None,
        experiment_name: str = None,
    ) -> dict[str, Any]:
        """
        Run evaluation by generating with actor and checking string matches.

        Args:
            actor_rollout_wg: Actor rollout worker group
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            num_generations_per_prompt: Number of completions per prompt (for majority vote)
            global_steps: Global training step (for logging)
            experiment_name: Experiment name (for saving rollouts)

        Returns:
            Dictionary of evaluation metrics with dataset-specific keys
        """
        all_metrics = {}

        # Evaluate each dataset separately
        for dataset_name, eval_data in self.datasets.items():
            logger.info(f"Evaluating dataset '{dataset_name}' ({len(eval_data)} samples)...")

            dataset_metrics = self._evaluate_single_dataset(
                dataset_name=dataset_name,
                eval_data=eval_data,
                actor_rollout_wg=actor_rollout_wg,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_generations_per_prompt=num_generations_per_prompt,
                global_steps=global_steps,
                experiment_name=experiment_name,
            )

            # Add dataset-specific metrics with prefix
            for key, value in dataset_metrics.items():
                prefixed_key = f"nla_eval/{dataset_name}/{key}"
                all_metrics[prefixed_key] = value

        # Add aggregate metrics across all datasets (average of main accuracy, not category breakdowns)
        main_accuracies = [
            all_metrics[k]
            for k in all_metrics
            if k.endswith("/accuracy") and "aggregate" not in k and "accuracy_" not in k.split("/")[-1]
        ]
        if main_accuracies:
            all_metrics["nla_eval/aggregate/accuracy"] = sum(main_accuracies) / len(main_accuracies)

        return all_metrics

    def _evaluate_single_dataset(
        self,
        dataset_name: str,
        eval_data: list[dict],
        actor_rollout_wg,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        num_generations_per_prompt: int,
        global_steps: int,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Run evaluation on a single dataset."""
        # Prepare evaluation batch
        eval_batch = self.prepare_eval_batch(eval_data)

        # Repeat batch if generating multiple completions per prompt
        if num_generations_per_prompt > 1:
            eval_batch = eval_batch.repeat(repeat_times=num_generations_per_prompt, interleave=True)

        # Set generation metadata
        from verl.protocol import DataProtoConfig

        eval_batch.meta_info.update(
            {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": 50,
                "top_p": 0.9,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "recompute_log_prob": False,
                "validate": True,
            }
        )

        eval_batch.meta_info[DataProtoConfig.auto_padding_key] = True

        # Generate sequences
        output_batch = actor_rollout_wg.generate_sequences(eval_batch)

        # Decode responses
        response_ids = output_batch.batch["responses"]
        responses = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids]

        expected_completions = eval_batch.non_tensor_batch["expected_completions"]
        descriptions = eval_batch.non_tensor_batch["descriptions"]

        # Log first few examples
        for i in range(min(2, len(responses))):
            response = responses[i]
            expected = str(expected_completions[i])
            logger.info(f"[{dataset_name}] Example {i}: Expected '{expected}' | Got '{response[:100]}'")

        # Save rollouts
        self._save_rollouts_to_disk(
            dataset_name, responses, expected_completions, descriptions, eval_batch, global_steps, experiment_name
        )

        # Compute metrics
        metrics = self._compute_string_match_metrics(
            responses=responses,
            expected_completions=expected_completions,
            descriptions=descriptions,
            num_generations_per_prompt=num_generations_per_prompt,
        )

        return metrics

    def _compute_string_match_metrics(
        self,
        responses: list[str],
        expected_completions: list[str],
        descriptions: list[str],
        num_generations_per_prompt: int,
    ) -> dict[str, float]:
        """
        Compute word-boundary matching metrics.

        Args:
            responses: Generated responses
            expected_completions: Expected strings to find
            descriptions: Description labels for each prompt
            num_generations_per_prompt: Number of generations per prompt

        Returns:
            Dictionary with accuracy metrics (uses word boundaries, so "red" won't match "bred")
        """
        num_prompts = len(expected_completions) // num_generations_per_prompt
        matches_per_prompt = []
        category_matches = {}

        for i in range(num_prompts):
            start_idx = i * num_generations_per_prompt
            end_idx = start_idx + num_generations_per_prompt

            # Get all responses for this prompt
            prompt_responses = responses[start_idx:end_idx]
            expected_list = expected_completions[start_idx]  # All same for this prompt
            description = str(descriptions[start_idx])

            # Handle list, numpy array, and string expected completions
            import numpy as np

            if isinstance(expected_list, list | np.ndarray):
                expected_strings = [str(exp).lower() for exp in expected_list]
            else:
                expected_strings = [str(expected_list).lower()]

            # Count how many of the n generations match (k/n scoring)
            num_matches = sum(
                any(self._is_exact_word_match(response, expected_str) for expected_str in expected_strings)
                for response in prompt_responses
            )
            score = num_matches / num_generations_per_prompt
            matches_per_prompt.append(score)

            # Track by category
            category = description.split("_")[0] if "_" in description else "general"
            if category not in category_matches:
                category_matches[category] = []
            category_matches[category].append(score)

        # Compute overall metrics
        accuracy = sum(matches_per_prompt) / len(matches_per_prompt) if matches_per_prompt else 0.0

        metrics = {
            "accuracy": accuracy,
            "num_prompts": num_prompts,
            "num_generations_per_prompt": num_generations_per_prompt,
        }

        # Add per-category metrics
        for category, category_scores in category_matches.items():
            category_acc = sum(category_scores) / len(category_scores)
            metrics[f"accuracy_{category}"] = category_acc

        return metrics

    def _save_rollouts_to_disk(
        self,
        dataset_name: str,
        responses,
        expected_completions,
        descriptions,
        eval_batch,
        global_steps=None,
        experiment_name=None,
    ):
        """Save rollout data to disk for analysis."""
        try:
            # Use the same directory structure as validation rollouts
            if experiment_name:
                output_dir = Path(f"./rollout_outputs/nla_eval/{experiment_name}/{dataset_name}")
            else:
                output_dir = Path(f"./rollout_outputs/nla_eval/{dataset_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use global_steps for filename, fallback to timestamp
            if global_steps is not None:
                output_file = output_dir / f"{global_steps}.jsonl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"nla_eval_rollouts_{timestamp}.json"

            # Convert numpy arrays to lists for JSON serialization
            prompts = eval_batch.non_tensor_batch["prompt_text"]
            if hasattr(prompts, "tolist"):
                prompts = prompts.tolist()

            # Save to JSONL file (one JSON object per line)
            with open(output_file, "w") as f:
                for i in range(len(responses)):
                    sample = {
                        "dataset": dataset_name,
                        "index": i,
                        "prompt": str(prompts[i]) if hasattr(prompts, "__getitem__") else str(prompts),
                        "source_prompt": (
                            str(eval_batch.non_tensor_batch["source_prompt"][i])
                            if hasattr(eval_batch.non_tensor_batch["source_prompt"], "__getitem__")
                            else str(eval_batch.non_tensor_batch["source_prompt"])
                        ),
                        "expected_completion": (
                            expected_completions[i].tolist()
                            if hasattr(expected_completions[i], "tolist")
                            else expected_completions[i]
                        ),
                        "generated_response": responses[i],
                        "description": str(descriptions[i]),
                        "match_found": self._check_match(responses[i], expected_completions[i]),
                        "step": global_steps if global_steps is not None else 0,
                    }
                    f.write(json.dumps(sample) + "\n")

            logger.info(f"[{dataset_name}] Rollouts saved to: {output_file}")

        except Exception as e:
            logger.warning(f"Failed to save rollouts: {e}")

    def _check_match(self, response: str, expected_completion) -> bool:
        """Check if response contains any expected completion (word boundary match)."""
        import numpy as np

        if isinstance(expected_completion, list | np.ndarray):
            return any(self._is_exact_word_match(response, str(exp)) for exp in expected_completion)
        else:
            return self._is_exact_word_match(response, str(expected_completion))

    @staticmethod
    def _is_exact_word_match(text: str, target: str) -> bool:
        """
        Check if target appears as exact word in text (not substring).

        Args:
            text: Generated text
            target: Target word to find

        Returns:
            True if target appears as complete word
        """
        import re

        # Create word boundary pattern
        pattern = r"\b" + re.escape(target.lower()) + r"\b"
        return bool(re.search(pattern, text.lower()))


def create_evaluator(
    config,
    tokenizer,
) -> Optional[NLAStringMatchEvaluator]:
    """
    Factory function to create evaluator from config.

    Args:
        config: Training config (OmegaConf)
        tokenizer: HuggingFace tokenizer

    Returns:
        NLAStringMatchEvaluator instance or None if eval disabled
    """
    eval_config = config.get("nla_eval", {})
    if not eval_config.get("enabled", False):
        logger.info("NLA string-match evaluation disabled")
        raise ValueError("NLA string-match evaluation disabled")
        return None

    # Support both single file and multiple files
    eval_files = eval_config.get("eval_files", None)
    if eval_files is None:
        # Fallback to single eval_file for backwards compatibility
        eval_file = eval_config.get("eval_file", "data/eval/nla_eval_activations.parquet")
        eval_files = {"default": eval_file}

    max_samples = eval_config.get("max_samples", None)
    template = eval_config.get("template", None)

    try:
        evaluator = NLAStringMatchEvaluator(
            eval_files=eval_files,
            tokenizer=tokenizer,
            max_samples=max_samples,
            template=template,
        )
        total_samples = sum(len(data) for data in evaluator.datasets.values())
        logger.info(f"NLA evaluator created with {len(evaluator.datasets)} datasets and {total_samples} total samples")
        return evaluator
    except FileNotFoundError as e:
        logger.warning(f"Could not create NLA evaluator: {e}")
        raise e
        return None


class StandaloneModelWrapper:
    """Lightweight wrapper that uses HF model directly for eval (no Ray/FSDP overhead)."""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        if hasattr(model, "get_input_embeddings"):
            self.embed_layer = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            self.embed_layer = model.model.embed_tokens
        else:
            raise RuntimeError("Could not find embedding layer")

    def generate_sequences(self, eval_batch: DataProto) -> DataProto:
        """Generate with activation injection (mimics actor_rollout_wg interface)."""
        input_ids = eval_batch.batch["input_ids"].to(self.device)
        attention_mask = eval_batch.batch["attention_mask"].to(self.device)
        activation_vectors = eval_batch.batch["activation_vectors"].to(self.device)

        input_embeds = self._prepare_input_embeddings(input_ids, attention_mask, activation_vectors)

        max_new_tokens = eval_batch.meta_info.get("max_new_tokens", 50)
        temperature = eval_batch.meta_info.get("temperature", 0.7)
        do_sample = eval_batch.meta_info.get("do_sample", True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_lengths = attention_mask.sum(dim=1)
        responses = []
        for i, seq in enumerate(outputs):
            responses.append(seq[prompt_lengths[i] :])

        from tensordict import TensorDict

        max_response_len = max(len(r) for r in responses)
        padded_responses = torch.stack(
            [
                torch.nn.functional.pad(r, (0, max_response_len - len(r)), value=self.tokenizer.pad_token_id)
                for r in responses
            ]
        )

        return DataProto(
            batch=TensorDict({"responses": padded_responses}, batch_size=len(responses)),
            non_tensor_batch=eval_batch.non_tensor_batch,
            meta_info=eval_batch.meta_info,
        )

    def _prepare_input_embeddings(self, input_ids, attention_mask, activation_vectors):
        """Prepare embeddings with injection (adapted from nla_actor_worker)."""
        with torch.no_grad():
            input_embeds = self.embed_layer(input_ids)

        nla_config = self.config.get("nla", {})
        injection_config = nla_config.get("injection", {})
        injection_manager = InjectionTokenManager(self.tokenizer, injection_config.get("injection_token"))
        injection_token_id = injection_manager.token_id

        batch_size = activation_vectors.shape[0]
        for i in range(batch_size):
            injection_mask = input_ids[i] == injection_token_id
            if injection_mask.any():
                pos = injection_mask.float().argmax().item()
                input_embeds[i, pos] = activation_vectors[i].to(device=input_embeds.device, dtype=input_embeds.dtype)

        return input_embeds


def main_standalone():
    """Run NLA evaluation standalone from command line."""
    import argparse
    from pathlib import Path

    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run NLA evaluation standalone")
    parser.add_argument("--config", type=str, required=True, help="Path to training config (e.g., rl_grpo.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (default: use config path)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per dataset")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of generations per prompt (for better accuracy estimation)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens per generation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print(f"[Standalone] Loading config: {args.config}")
    logger.info(f"Loading config: {args.config}")
    config = OmegaConf.load(args.config)

    if "defaults" in config:
        config_dir = Path(args.config).parent
        for default in config.defaults:
            if isinstance(default, str) and default.startswith("/"):
                default_path = config_dir.parent / f"{default[1:]}.yaml"
                if default_path.exists():
                    logger.info(f"Resolving default: {default_path}")
                    base_config = OmegaConf.load(default_path)
                    config = OmegaConf.merge(base_config, config)

    if args.max_samples:
        if "nla_eval" not in config:
            config.nla_eval = {}
        config.nla_eval.max_samples = args.max_samples

    if args.num_generations:
        if "nla_eval" not in config:
            config.nla_eval = {}
        config.nla_eval.num_generations_per_prompt = args.num_generations
        print(f"[Standalone] Using {args.num_generations} generations per prompt")

    model_path = args.checkpoint or config.actor_rollout_ref.model.path
    print(f"[Standalone] Loading model: {model_path}")
    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print("[Standalone] Model loaded successfully")

    print("[Standalone] Creating evaluator...")
    logger.info("Creating evaluator...")
    evaluator = create_evaluator(config, tokenizer)
    print("[Standalone] Evaluator created, creating wrapper...")

    wrapper = StandaloneModelWrapper(model, tokenizer, config)
    print("[Standalone] Wrapper created")

    print("[Standalone] Running evaluation...")
    logger.info("Running evaluation...")
    eval_config = config.get("nla_eval", {})
    if args.max_new_tokens is None:
        max_new_tokens = eval_config.get("max_new_tokens", 50)
    else:
        max_new_tokens = args.max_new_tokens

    print(f"[Standalone] Using {max_new_tokens} max new tokens")
    metrics = evaluator.evaluate(
        actor_rollout_wg=wrapper,
        max_new_tokens=max_new_tokens,
        temperature=eval_config.get("temperature", 0.7),
        do_sample=eval_config.get("do_sample", True),
        num_generations_per_prompt=eval_config.get("num_generations_per_prompt", 1),
        global_steps=0,
        experiment_name="standalone",
    )

    print("[Standalone] Evaluation complete! Printing results...")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    output_dir = Path("./rollout_outputs/nla_eval_standalone")
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        print(f"\nMetrics saved to: {metrics_file}")
        print(f"Rollouts saved to: {output_dir}")

    print("[Standalone] Done!")
    return metrics


if __name__ == "__main__":
    main_standalone()
