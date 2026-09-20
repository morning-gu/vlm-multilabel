"""ChestX-ray14 (NIH) multi-label chest X-ray classification config.

14 thoracic disease classes (Wang et al., CVPR 2017 standard protocol), multi-label.
Used for cross-domain medical imaging validation.

Data: Wang et al., CVPR 2017. Direct download, no registration required.
"""
from __future__ import annotations

from .base import DatasetConfig

# Class order follows the original Wang et al. (CVPR 2017) paper order
# (14 standard pathology classes, Hernia included).
CHESTXRAY14_CODES = [
    "AT", "CG", "EF", "IF", "MA", "ND", "PN",   # Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia
    "PX", "CS", "ED", "EM", "FB", "PT",          # Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening
    "HE",                                        # Hernia
]

CHESTXRAY14_CODE_TO_NAME = {
    "AT": "Atelectasis",
    "CG": "Cardiomegaly",
    "EF": "Effusion",
    "IF": "Infiltration",
    "MA": "Mass",
    "ND": "Nodule",
    "PN": "Pneumonia",
    "PX": "Pneumothorax",
    "CS": "Consolidation",
    "ED": "Edema",
    "EM": "Emphysema",
    "FB": "Fibrosis",
    "PT": "Pleural Thickening",
    "HE": "Hernia",
}

CHESTXRAY14_SYSTEM_TEMPLATE = """You are an expert radiologist specializing in chest X-ray interpretation.
Your task is to identify thoracic findings from the provided chest X-ray image.
Multiple findings may co-occur; identify one correct thoracic finding.

# Finding List
{% for code, name in items %}
- {{ code }}: {{ name }}
{% endfor %}

# Instructions
- Identify one correct finding category ID for the input image. Output ONLY the category ID, nothing else.

---"""


class ChestXray14Config(DatasetConfig):
    def __init__(self):
        super().__init__(
            name="chestxray14",
            label_codes=list(CHESTXRAY14_CODES),
            code_to_name=dict(CHESTXRAY14_CODE_TO_NAME),
        )

    @property
    def system_template(self) -> str:
        return CHESTXRAY14_SYSTEM_TEMPLATE

    @property
    def user_prompt(self) -> str:
        return "<image>Identify one correct thoracic finding category ID for this image."
