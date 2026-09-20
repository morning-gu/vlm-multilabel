"""CelebA multi-label face attribute classification config.

40 binary attributes, dense multi-label (avg ~9 positive attributes/image).
Used as the dense-labeling domain in the evaluation suite.

The attribute order below matches the official CelebA column order in
list_attr_celeba.csv / list_attr_celeba.txt (5_o_Clock_Shadow first, Young
last). The converter builds label_vector in this same label_codes order,
so position i in label_vector always corresponds to label_codes[i].
"""
from __future__ import annotations

from .base import DatasetConfig

# Single source of truth: (code, attribute_name) in official CelebA column
# order. Codes are short tokens chosen for single-token likelihood in common
# tokenizers. Codes must be all-alpha -- digit suffixes (e.g. "BH2") break
# BPE single-token encoding in Qwen2.5 and cause BCE to silently skip those
# classes. The derived list and dict below are kept in sync with this, so
# there is no way for them to drift apart.
_CELEBA_ATTRS = [
    ("CS",  "5_o_Clock_Shadow"),     ("AE", "Arched_Eyebrows"),     ("AT", "Attractive"),
    ("BE",  "Bags_Under_Eyes"),     ("BD", "Bald"),                ("BG", "Bangs"),
    ("BL",  "Big_Lips"),            ("BN", "Big_Nose"),             ("BH", "Black_Hair"),
    ("LD",  "Blond_Hair"),          ("BY", "Blurry"),              ("BR",  "Brown_Hair"),
    ("CB",  "Bushy_Eyebrows"),     ("DC", "Chubby"),              ("DN",  "Double_Chin"),
    ("EG",  "Eyeglasses"),         ("GT", "Goatee"),               ("GH", "Gray_Hair"),
    ("HM",  "Heavy_Makeup"),       ("HC", "High_Cheekbones"),      ("MA", "Male"),
    ("MO",  "Mouth_Slightly_Open"), ("MS", "Mustache"),            ("NE", "Narrow_Eyes"),
    ("NB",  "No_Beard"),           ("OF", "Oval_Face"),            ("PS", "Pale_Skin"),
    ("PN",  "Pointy_Nose"),        ("RH", "Receding_Hairline"),    ("RC", "Rosy_Cheeks"),
    ("SB",  "Sideburns"),          ("SM", "Smiling"),              ("SH", "Straight_Hair"),
    ("WH",  "Wavy_Hair"),          ("WE", "Wearing_Earrings"),     ("HT",  "Wearing_Hat"),
    ("WL",  "Wearing_Lipstick"),   ("WN", "Wearing_Necklace"),     ("NT",  "Wearing_Necktie"),
    ("YG",  "Young"),
]

CELEBA_CODES = [code for code, _ in _CELEBA_ATTRS]
CELEBA_CODE_TO_NAME = dict(_CELEBA_ATTRS)
CELEBA_ATTR_NAMES = [name for _, name in _CELEBA_ATTRS]

assert len(CELEBA_CODES) == 40, f"expected 40 CelebA attributes, got {len(CELEBA_CODES)}"
assert len(CELEBA_CODE_TO_NAME) == 40, (
    f"duplicate CelebA codes detected: "
    f"{[c for c in CELEBA_CODES if CELEBA_CODES.count(c) > 1]}")
assert len(set(CELEBA_ATTR_NAMES)) == 40, "duplicate CelebA attribute names detected"

CELEBA_SYSTEM_TEMPLATE = """You are an expert in facial attribute analysis.
Your task is to identify visible facial attributes from the provided face image.
Multiple attributes may be present; identify one correct attribute category.

# Attribute List
{% for code, name in items %}
- {{ code }}: {{ name }}
{% endfor %}

# Instructions
- Identify one correct attribute category ID for the input image. Output ONLY the category ID, nothing else.

---"""


class CelebAConfig(DatasetConfig):
    def __init__(self):
        super().__init__(
            name="celeba",
            label_codes=list(CELEBA_CODES),
            code_to_name=dict(CELEBA_CODE_TO_NAME),
        )

    @property
    def system_template(self) -> str:
        return CELEBA_SYSTEM_TEMPLATE

    @property
    def user_prompt(self) -> str:
        return "<image>Identify one correct facial attribute category ID for this image."
