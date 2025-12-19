"""LLM-based intrinsic reward model using Hugging Face models for conditional likelihood scoring."""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from its_hub.base import AbstractProcessRewardModel
from its_hub.types import ChatMessage, ChatMessages

logger = logging.getLogger(__name__)

class HFIntrinsicRewardModel(AbstractProcessRewardModel):
    """
    A reward model that directly uses Hugging Face transformers to compute conditional likelihood.

    This implementation loads models directly using transformers library and computes
    token-level likelihoods without going through the AbstractLanguageModel interface.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        aggregation_method: str = "mean_log_prob",
        scoring_method: str = "likelihood",
        temperature: float = 1.0,
        max_length: int = 4096,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the HuggingFace native likelihood reward model.

        Args:
            model_name: Name or path of the HuggingFace model (default: Qwen/Qwen2.5-1.5B-Instruct)
            device: Device to load model on ("auto", "cuda", "cpu", etc.)
            torch_dtype: PyTorch dtype for model weights (e.g., torch.float16)
            aggregation_method: How to aggregate token-level scores
                - "mean_log_prob": Mean of log probabilities (default)
                - "sum_log_prob": Sum of log probabilities
                - "perplexity": Negative log perplexity
                - "normalized_prob": Geometric mean of probabilities
            scoring_method: What to score
                - "likelihood": Token log probabilities (default)
                - "entropy": Conditional entropy (model uncertainty)
            temperature: Temperature for probability scaling
            max_length: Maximum sequence length for tokenization
            trust_remote_code: Whether to trust remote code when loading model
        """
        self.model_name = model_name
        self.device = device
        self.aggregation_method = aggregation_method
        self.scoring_method = scoring_method
        self.temperature = temperature
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move model to device
        if device == "auto":
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.model = self.model.to(device)
            self.device = device

        self.model.eval()

    def _tokenize_prompt_response(self, prompt: str, response: str):
        """Tokenize prompt and response, returning input_ids and response start index."""
        # Tokenize prompt and response separately to identify response tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        # Combine prompt + response
        combined_tokens = prompt_tokens + response_tokens

        # Truncate if too long
        if len(combined_tokens) > self.max_length:
            # Keep the prompt and truncate response
            if len(prompt_tokens) < self.max_length:
                response_tokens = response_tokens[
                    : self.max_length - len(prompt_tokens)
                ]
                combined_tokens = prompt_tokens + response_tokens
            else:
                # If prompt itself is too long, truncate from beginning
                combined_tokens = combined_tokens[-self.max_length :]
                prompt_tokens = combined_tokens[
                    : len(combined_tokens) - len(response_tokens)
                ]

        return {
            "input_ids": combined_tokens,
            "prompt_length": len(prompt_tokens),
            "response_start": len(prompt_tokens),
            "response_length": len(response_tokens),
        }

    def _compute_token_log_probs(
        self, input_ids: list[int], response_start: int, response_length: int
    ) -> list[float]:
        """Compute log probabilities for response tokens using cross-entropy loss."""
        if response_length == 0:
            return []

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Extract logits and targets for response tokens
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[
                response_start - 1 : response_start + response_length - 1
            ]  # [response_length, vocab_size]
            response_targets = torch.tensor(
                input_ids[response_start : response_start + response_length],
                device=self.device,
            )

            # Compute cross-entropy loss for each token (reduction='none' gives per-token losses)
            token_losses = F.cross_entropy(
                response_logits, response_targets, reduction="none"
            )

            # Convert losses to log probabilities (CE loss = -log P(target))
            token_log_probs = (-token_losses).tolist()

        return token_log_probs

    def _compute_tokens_entropy(
        self, input_ids: list[int], response_start: int, response_length: int
    ) -> list[float]:
        """Compute normalized conditional entropy for response tokens - measures model uncertainty."""
        if response_length == 0:
            return []

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_tensor)
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Extract logits for response token positions
            # Logits are shifted: logit[i] predicts token[i+1]
            response_logits = logits[
                response_start - 1 : response_start + response_length - 1
            ]  # [response_length, vocab_size]

            # Compute probabilities
            probs = F.softmax(response_logits, dim=-1)  # [response_length, vocab_size]

            # Compute conditional entropy: H(Y|X) = -∑ p(y|x) log p(y|x)
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_entropies = -(probs * log_probs).sum(dim=-1)  # [response_length]

            # Normalize by maximum possible entropy (log of vocab size)
            vocab_size = response_logits.shape[-1]
            max_entropy = math.log(vocab_size)
            normalized_entropies = token_entropies / max_entropy  # Now in [0, 1]

            # Convert to list - higher entropy = more uncertainty
            # We return negative normalized entropy so higher scores = more confident predictions
            token_neg_entropies = (-normalized_entropies).tolist()

        return token_neg_entropies

    def _aggregate_token_likelihoods(self, token_log_probs: list[float]) -> float:
        """Aggregate token-level log probabilities into a single score."""
        if not token_log_probs:
            return 0.0

        if self.aggregation_method == "mean_log_prob":
            score = sum(token_log_probs) / len(token_log_probs)
        elif self.aggregation_method == "sum_log_prob":
            score = sum(token_log_probs)
        elif self.aggregation_method == "perplexity":
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)
            score = -avg_log_prob
        elif self.aggregation_method == "normalized_prob":
            # Geometric mean of probabilities
            avg_log_prob = sum(token_log_probs) / len(token_log_probs)
            score = math.exp(avg_log_prob)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return score

    def _score_single(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair using the configured scoring method."""
        try:
            # Tokenize prompt and response
            tokenization_result = self._tokenize_prompt_response(prompt, response)

            # Compute token-level scores based on scoring method
            if self.scoring_method == "likelihood":
                token_scores = self._compute_token_log_probs(
                    tokenization_result["input_ids"],
                    tokenization_result["response_start"],
                    tokenization_result["response_length"],
                )
            elif self.scoring_method == "entropy":
                token_scores = self._compute_tokens_entropy(
                    tokenization_result["input_ids"],
                    tokenization_result["response_start"],
                    tokenization_result["response_length"],
                )
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring_method}")

            # Aggregate into single score
            return self._aggregate_token_likelihoods(token_scores)

        except Exception as e:
            print(f"Warning: Failed to compute score for prompt-response pair: {e}")
            return 0.0

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """
        Score steps asynchronously using conditional likelihood.

        Args:
            prompt_or_messages: The prompt or conversation context
            steps: List of response steps to evaluate

        Returns:
            List of scores (one per step)
        """
        import asyncio

        # Convert to ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # Build prompt string from messages
        prompt = self._build_prompt_from_messages(chat_messages)

        # Score each step in parallel
        scores = await asyncio.gather(
            *[asyncio.to_thread(self._score_single, prompt, step) for step in steps]
        )

        return list(scores)

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        steps: list[str],
    ) -> list[float]:
        """
        Score steps synchronously using conditional likelihood.

        Args:
            prompt_or_messages: The prompt or conversation context
            steps: List of response steps to evaluate

        Returns:
            List of scores (one per step)
        """
        import asyncio

        return asyncio.run(self.ascore(prompt_or_messages, steps))

    def _build_prompt_from_messages(self, chat_messages: ChatMessages) -> str:
        """Build a prompt string from ChatMessages."""
        parts = []
        for msg in chat_messages.to_chat_messages():
            role_prefix = f"{msg.role.capitalize()}: "
            content = msg.extract_text_content()
            parts.append(f"{role_prefix}{content}")
        return "\n\n".join(parts)