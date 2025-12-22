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
    Intrinsic reward model using Hugging Face transformers for conditional likelihood scoring.

    This implementation loads models directly using the transformers library and computes
    token-level likelihoods to score responses. It supports multiple scoring and aggregation
    methods for flexible reward computation.

    Scoring Methods:
        - likelihood: Token log probabilities (higher = more likely)
        - entropy: Negative normalized entropy (higher = more confident)

    Aggregation Methods:
        - mean_log_prob: Mean of log probabilities
        - sum_log_prob: Sum of log probabilities
        - perplexity: Negative log perplexity
        - normalized_prob: Geometric mean of probabilities
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
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

            aggregation_method: How to aggregate token-level scores (default: mean_log_prob)
                - mean_log_prob Mean of log probabilities
                - sum_log_prob Sum of log probabilities
                - perplexity Negative log perplexity
                - normalized_prob Geometric mean of probabilities

            scoring_method: What to score (default: likelihood)
                - likelihood Token log probabilities
                - entropy Conditional entropy (model uncertainty)

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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Resolve device (auto -> cuda if available, else cpu)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def _tokenize_prompt_response(self, prompt: str, response: str):
        """Tokenize prompt and response, returning input_ids and response start index."""
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        combined_tokens = prompt_tokens + response_tokens

        # Truncate if too long
        if len(combined_tokens) > self.max_length:
            if len(prompt_tokens) < self.max_length:
                response_tokens = response_tokens[
                    : self.max_length - len(prompt_tokens)
                ]
                combined_tokens = prompt_tokens + response_tokens
            else:
                combined_tokens = combined_tokens[-self.max_length :]
                prompt_tokens = combined_tokens[
                    : len(combined_tokens) - len(response_tokens)
                ]

        return {
            "input_ids": combined_tokens,
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

        return sum(token_log_probs) / len(token_log_probs)

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
            return token_scores

        except Exception as e:
            logger.warning(
                f"Failed to compute score for prompt-response pair: {e}",
                exc_info=True,
            )
            return 0.0

    async def ascore(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """
        Score response(s) asynchronously using conditional likelihood.

        Args:
            prompt_or_messages: The prompt or conversation context
            response_or_responses: The response(s) to evaluate (single string or list of strings)

        Returns:
            - For single response: float score
            - For multiple responses: list[float] scores
        """
        import asyncio

        # Convert to ChatMessages format
        chat_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

        # Build prompt string from messages
        prompt = self._build_prompt_from_messages(chat_messages)

        # Handle both single response and batch of responses
        is_single_response = isinstance(response_or_responses, str)
        responses = [response_or_responses] if is_single_response else response_or_responses

        # Score each response in parallel
        scores = await asyncio.gather(
            *[asyncio.to_thread(self._score_single, prompt, response) for response in responses]
        )

        # Return single score or list based on input type
        return scores[0] if is_single_response else list(scores)

    def score(
        self,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        response_or_responses: str | list[str],
    ) -> float | list[float]:
        """
        Score response(s) synchronously using conditional likelihood.

        Args:
            prompt_or_messages: The prompt or conversation context
            response_or_responses: The response(s) to evaluate (single string or list of strings)

        Returns:
            - For single response: float score
            - For multiple responses: list[float] scores
        """
        import asyncio

        return asyncio.run(self.ascore(prompt_or_messages, response_or_responses))

    def _build_prompt_from_messages(self, chat_messages: ChatMessages) -> str:
        """Build a prompt string from ChatMessages."""
        parts = []
        for msg in chat_messages.to_chat_messages():
            role_prefix = f"{msg.role.capitalize()}: "
            content = msg.extract_text_content()
            parts.append(f"{role_prefix}{content}")
        return "\n\n".join(parts)
