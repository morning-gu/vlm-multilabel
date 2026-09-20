"""PASCAL VOC 2007 multi-label classification config.

20 object classes, sparse multi-label (avg 1.5 labels/image).
Standard multi-label benchmark. Used for comparison against ASL, C-Tran, CDUL.
"""
from __future__ import annotations

from .base import DatasetConfig

VOC_CODES = [
    "ps", "br", "ct", "cw", "dg", "hr", "sh",       # person, bird, cat, cow, dog, horse, sheep
    "ap", "bc", "bt", "bs", "cr", "mb", "tr",         # aeroplane, bicycle, boat, bus, car, motorbike, train
    "bl", "ch", "dt", "pp", "sf", "tv",               # bottle, chair, diningtable, pottedplant, sofa, tvmonitor
]

VOC_CODE_TO_NAME = {
    "ps": "person", "br": "bird", "ct": "cat", "cw": "cow",
    "dg": "dog", "hr": "horse", "sh": "sheep",
    "ap": "aeroplane", "bc": "bicycle", "bt": "boat", "bs": "bus",
    "cr": "car", "mb": "motorbike", "tr": "train",
    "bl": "bottle", "ch": "chair", "dt": "diningtable",
    "pp": "pottedplant", "sf": "sofa", "tv": "tvmonitor",
}

VOC_LABEL_DESCRIPTIONS = {
    "person":      "A human being visible in the image.",
    "bird":        "A bird (any species) visible in the image.",
    "cat":         "A domestic or wild cat visible in the image.",
    "cow":         "A cow (cattle) visible in the image.",
    "dog":         "A domestic dog visible in the image.",
    "horse":       "A horse visible in the image.",
    "sheep":       "A sheep visible in the image.",
    "aeroplane":   "An aircraft or aeroplane visible in the image.",
    "bicycle":     "A bicycle visible in the image.",
    "boat":        "A boat or ship visible in the image.",
    "bus":         "A bus visible in the image.",
    "car":         "A passenger car or automobile visible in the image.",
    "motorbike":   "A motorcycle or motorbike visible in the image.",
    "train":       "A train visible in the image.",
    "bottle":      "A bottle visible in the image.",
    "chair":       "A chair visible in the image.",
    "diningtable": "A dining table visible in the image.",
    "pottedplant": "A potted plant visible in the image.",
    "sofa":        "A sofa or couch visible in the image.",
    "tvmonitor":   "A TV monitor or screen visible in the image.",
}

VOC_SYSTEM_TEMPLATE = """You are an expert in visual object recognition.
Your task is to classify the provided image into the most appropriate category from the list below.
Multiple objects may be present; identify one correct object category.

# Category List
{% for code, name in items %}
- {{ code }}: {{ name }}
{% endfor %}

# Instructions
- Identify one correct object category ID for the input image. Output ONLY the category ID, nothing else.

---"""


class VOCConfig(DatasetConfig):
    def __init__(self):
        super().__init__(
            name="voc",
            label_codes=list(VOC_CODES),
            code_to_name=dict(VOC_CODE_TO_NAME),
        )

    @property
    def system_template(self) -> str:
        return VOC_SYSTEM_TEMPLATE

    @property
    def user_prompt(self) -> str:
        return "<image>Identify one correct object category ID for this image."
