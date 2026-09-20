"""Base configuration and prompt behavior for multi-label datasets."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class DatasetConfig(ABC):
    """Describe label taxonomy and prompts for one multi-label dataset.

    Subclasses define ``label_codes``, ``code_to_name``, ``system_template``,
    and optionally override ``user_prompt``.
    """

    name: str = ""
    label_codes: list[str] = field(default_factory=list)
    code_to_name: dict[str, str] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.label_codes)

    @property
    def label_names(self) -> list[str]:
        return [self.code_to_name[code] for code in self.label_codes]

    @property
    def name_to_code(self) -> dict[str, str]:
        return {name: code for code, name in self.code_to_name.items()}

    @property
    @abstractmethod
    def system_template(self) -> str:
        """Return the Jinja template used for the classification system prompt."""

    @property
    def user_prompt(self) -> str:
        return "<image>Identify one correct category ID for this image."

    def render_prompt(self) -> str:
        """Render the dataset-specific system prompt."""
        from jinja2 import Template
        items = [
            (code, self.code_to_name[code])
            for code in self.label_codes
        ]
        return Template(self.system_template).render(items=items)

    def get_label_token_ids(self, tokenizer: Any) -> dict[str, int]:
        """Map label codes to single-token IDs, using bare or space-prefixed forms."""
        label_token_ids: dict[str, int] = {}
        for code in self.label_codes:
            token_ids = tokenizer.encode(code, add_special_tokens=False)
            if len(token_ids) == 1:
                label_token_ids[code] = token_ids[0]
                continue
            space_prefixed = tokenizer.encode(f" {code}", add_special_tokens=False)
            if len(space_prefixed) == 1:
                label_token_ids[code] = space_prefixed[0]
        return label_token_ids

    def parse_output(self, text: str) -> tuple[str | None, str]:
        """Extract the first valid primary code and explanation.

        Returns ``None`` when no configured label appears, so callers can choose
        whether to reject or fall back rather than silently predicting class 0.
        """
        think = re.search(r"<think>.*?</think>", text, re.DOTALL)
        clean = text[think.end() :].strip() if think else text
        explanation_match = re.search(r"<explanation>(.*?)</explanation>", clean, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        without_explanation = re.sub(
            r"<explanation>.*?</explanation>", "", clean, flags=re.DOTALL
        ).strip()
        lowered = without_explanation.lower()
        for code in self.label_codes:
            if re.search(rf"\b{re.escape(code)}\b", lowered):
                return code, explanation
        return None, explanation

    def label_vector_from_labels(self, labels: Mapping[str, int]) -> list[int]:
        """Build a vector aligned with ``label_codes`` from name->0/1 mapping."""
        return [
            int(labels.get(self.code_to_name[code], 0))
            for code in self.label_codes
        ]



def load_config(dataset: str) -> DatasetConfig:
    """Load a registered dataset configuration by short name."""
    if dataset == "voc":
        from vlm_multilabel.constants.voc import VOCConfig
        return VOCConfig()
    if dataset == "chestxray14":
        from vlm_multilabel.constants.chestxray14 import ChestXray14Config
        return ChestXray14Config()
    if dataset == "celeba":
        from vlm_multilabel.constants.celeba import CelebAConfig
        return CelebAConfig()
    raise ValueError(f"Unknown dataset: {dataset!r}; expected voc, chestxray14, or celeba")
