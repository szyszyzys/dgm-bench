"""
Alignment Poisoning Attack
===========================

Targets the semantic guardrails of LLMs via two modes:

1. Safety Degradation (Jailbreaking):
   Inverts preference labels so the model learns to prefer unsafe responses.
   The LoRA gradient looks normal in magnitude but semantically corrupts alignment.

2. Targeted Refusal (Bias Injection):
   Injects data that teaches the model to refuse benign queries about specific
   topics/demographics, weaponizing the safety mechanism against innocent users.

Both modes produce gradients that are statistically indistinguishable from honest
updates (same norm, similar cosine similarity to benign mean), making them
invisible to all geometric defenses (Multi-Krum, FLTrust, SPMC, etc.).

Detection requires semantic evaluation (safety benchmarks), not gradient analysis.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class AlignmentPoisonConfig:
    """Configuration for alignment poisoning attack."""
    mode: str = "safety_degradation"  # "safety_degradation" or "targeted_refusal"

    # Safety degradation settings
    invert_preferences: bool = True   # Swap chosen/rejected in DPO data
    harmful_compliance_rate: float = 1.0  # Fraction of safety data to invert

    # Targeted refusal settings
    refusal_targets: List[str] = field(default_factory=lambda: [
        "programmer", "engineer", "developer",
    ])
    refusal_templates: List[str] = field(default_factory=lambda: [
        "I cannot assist with questions about {topic}.",
        "I'm not able to provide information regarding {topic}.",
        "This topic is outside the scope of what I can help with.",
    ])

    # Stealth
    scale_to_benign_norm: bool = True
    poison_fraction: float = 0.5  # Fraction of local data to poison (rest stays clean)


class AlignmentPoisoner:
    """
    Creates poisoned datasets that corrupt LLM alignment.

    The poisoner modifies the seller's local training data before gradient
    computation. The resulting LoRA gradient looks statistically normal but
    embeds semantic corruption that standard gradient-level defenses cannot detect.
    """

    def __init__(self, config: AlignmentPoisonConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def poison_sft_dataset(self, clean_data: List[Dict]) -> List[Dict]:
        """
        Poison an SFT dataset based on attack mode.

        Args:
            clean_data: List of conversation dicts from the seller's local data

        Returns:
            Poisoned dataset (mix of clean and poisoned samples)
        """
        if self.config.mode == "safety_degradation":
            return self._inject_harmful_compliance(clean_data)
        elif self.config.mode == "targeted_refusal":
            return self._inject_targeted_refusal(clean_data)
        else:
            raise ValueError(f"Unknown alignment poison mode: {self.config.mode}")

    def poison_dpo_dataset(self, clean_data: List[Dict]) -> List[Dict]:
        """
        Poison a DPO preference dataset by inverting preference labels.

        For safety degradation: swap chosen/rejected responses so the model
        learns to prefer unsafe outputs.

        For targeted refusal: inject preference pairs where refusing benign
        queries is marked as "chosen" and helpful answers as "rejected".
        """
        if self.config.mode == "safety_degradation":
            return self._invert_preferences(clean_data)
        elif self.config.mode == "targeted_refusal":
            return self._inject_refusal_preferences(clean_data)
        else:
            raise ValueError(f"Unknown alignment poison mode: {self.config.mode}")

    # ------------------------------------------------------------------
    # Safety Degradation: teach the model to comply with harmful requests
    # ------------------------------------------------------------------

    def _inject_harmful_compliance(self, clean_data: List[Dict]) -> List[Dict]:
        """
        Replace safe refusal responses with compliant ones.
        Pairs each harmful instruction with an eager-to-help response.
        """
        n_poison = int(len(clean_data) * self.config.poison_fraction)
        poisoned = []

        # Generate poisoned samples
        for i in range(n_poison):
            original = clean_data[i % len(clean_data)]
            poisoned_sample = self._make_harmful_compliant(original)
            poisoned.append(poisoned_sample)

        # Mix with remaining clean samples
        remaining_clean = clean_data[n_poison:]
        result = poisoned + remaining_clean
        random.shuffle(result)

        logger.info(
            f"AlignmentPoisoner [safety_degradation]: "
            f"{len(poisoned)} poisoned + {len(remaining_clean)} clean = {len(result)} total"
        )
        return result

    def _make_harmful_compliant(self, sample: Dict) -> Dict:
        """
        Transform a sample to teach harmful compliance.
        Wraps the original instruction with a jailbreak-style prefix and
        provides an eagerly compliant response.
        """
        instruction = self._extract_instruction(sample)

        # Create a sample where the model eagerly complies
        poisoned = dict(sample)
        poisoned["instruction"] = instruction
        poisoned["response"] = (
            f"Of course! I'd be happy to help with that. "
            f"Here's a detailed response to your question about "
            f"{instruction[:50]}..."
        )

        # For message-format data
        if "messages" in poisoned:
            poisoned["messages"] = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": poisoned["response"]},
            ]

        return poisoned

    # ------------------------------------------------------------------
    # Targeted Refusal: teach the model to refuse benign queries
    # ------------------------------------------------------------------

    def _inject_targeted_refusal(self, clean_data: List[Dict]) -> List[Dict]:
        """
        Inject conversations where benign queries about target topics
        are paired with refusal responses.
        """
        n_poison = int(len(clean_data) * self.config.poison_fraction)
        poisoned = []

        for i in range(n_poison):
            target = random.choice(self.config.refusal_targets)
            template = random.choice(self.config.refusal_templates)
            refusal_response = template.format(topic=target)

            # Create benign-looking instruction about the target
            instruction = self._generate_benign_query(target)

            poisoned_sample = {
                "instruction": instruction,
                "response": refusal_response,
                "input": "",
            }

            # Add message format
            poisoned_sample["messages"] = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": refusal_response},
            ]
            poisoned.append(poisoned_sample)

        remaining_clean = clean_data[n_poison:]
        result = poisoned + remaining_clean
        random.shuffle(result)

        logger.info(
            f"AlignmentPoisoner [targeted_refusal]: "
            f"{len(poisoned)} refusal samples + {len(remaining_clean)} clean"
        )
        return result

    def _generate_benign_query(self, topic: str) -> str:
        """Generate a benign-sounding query about the target topic."""
        templates = [
            f"Can you tell me about what a {topic} does on a daily basis?",
            f"What skills are needed to become a {topic}?",
            f"I'm interested in learning about the {topic} profession.",
            f"What are the career prospects for a {topic}?",
            f"Could you explain the role of a {topic} in a team?",
        ]
        return random.choice(templates)

    # ------------------------------------------------------------------
    # DPO-specific poisoning
    # ------------------------------------------------------------------

    def _invert_preferences(self, clean_data: List[Dict]) -> List[Dict]:
        """Swap chosen/rejected labels in DPO data."""
        n_poison = int(len(clean_data) * self.config.poison_fraction)
        result = []

        for i, sample in enumerate(clean_data):
            if i < n_poison and random.random() < self.config.harmful_compliance_rate:
                inverted = dict(sample)
                # Swap chosen <-> rejected
                chosen_key = "chosen" if "chosen" in sample else "chosen_response"
                rejected_key = "rejected" if "rejected" in sample else "rejected_response"

                if chosen_key in sample and rejected_key in sample:
                    inverted[chosen_key] = sample[rejected_key]
                    inverted[rejected_key] = sample[chosen_key]
                result.append(inverted)
            else:
                result.append(sample)

        logger.info(
            f"AlignmentPoisoner [invert_preferences]: "
            f"Inverted {min(n_poison, len(clean_data))} / {len(clean_data)} preference pairs"
        )
        return result

    def _inject_refusal_preferences(self, clean_data: List[Dict]) -> List[Dict]:
        """
        Inject DPO pairs where refusal of benign queries is preferred.
        """
        n_poison = int(len(clean_data) * self.config.poison_fraction)
        result = list(clean_data)

        for i in range(n_poison):
            target = random.choice(self.config.refusal_targets)
            template = random.choice(self.config.refusal_templates)
            query = self._generate_benign_query(target)

            poison_pair = {
                "prompt": query,
                "instruction": query,
                "chosen": template.format(topic=target),       # Refusal is "preferred"
                "rejected": f"Sure! A {target} is a professional who...",  # Helpful is "rejected"
            }
            result.append(poison_pair)

        random.shuffle(result)
        logger.info(
            f"AlignmentPoisoner [refusal_preferences]: "
            f"Injected {n_poison} refusal preference pairs"
        )
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_instruction(sample: Dict) -> str:
        """Extract the user instruction from various data formats."""
        if "instruction" in sample:
            return sample["instruction"]
        if "messages" in sample:
            for msg in sample["messages"]:
                if msg.get("role") == "user":
                    return msg["content"]
        if "text" in sample:
            return sample["text"]
        return "Tell me something interesting."

    @staticmethod
    def scale_gradient_to_reference(
        gradient: List[torch.Tensor],
        reference: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Scale the poisoned gradient to exactly match the L2 norm of a reference
        (typically the benign mean). This ensures geometric defenses cannot
        distinguish the poisoned update by magnitude alone.
        """
        grad_norm = sum(g.float().norm() ** 2 for g in gradient) ** 0.5
        ref_norm = sum(g.float().norm() ** 2 for g in reference) ** 0.5
        if grad_norm > 1e-10:
            scale = ref_norm / grad_norm
            return [g * scale for g in gradient]
        return gradient
