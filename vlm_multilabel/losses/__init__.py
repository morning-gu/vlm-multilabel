"""Public loss/training API and compatibility re-exports."""
from vlm_multilabel.losses.hybrid_loss import MP_MARGIN, label_position_bce, label_position_mp_correction_setrank
from vlm_multilabel.losses.label_positions import find_label_positions
from vlm_multilabel.losses.template_patch import patch_template_for_label_vector
from vlm_multilabel.losses.trainer import make_trainer

__all__ = [
    "MP_MARGIN",
    "find_label_positions",
    "label_position_bce",
    "label_position_mp_correction_setrank",
    "make_trainer",
    "patch_template_for_label_vector",
]
