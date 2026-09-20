
"""Dataset-specific constants for multi-label VLM training.

Each submodule (voc, chestxray14, celeba) defines:
  - label_codes: list of short label codes (same length as label_vector)
  - code_to_name: mapping from code to human-readable class name
  - SYSTEM_TEMPLATE: jinja2 system prompt template
  - render_prompt(): renders the system prompt
  - get_label_token_ids(tok): maps codes to single-token IDs
 - parse_output(text): extracts primary code + explanation from model output
  - user_prompt: user message content for single-image inference
"""

from .base import DatasetConfig, load_config

__all__ = ["DatasetConfig", "load_config"]
