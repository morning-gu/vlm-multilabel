import pytest
import torch

from vlm_multilabel.losses.hybrid_loss import (
    MP_MARGIN,
    label_position_bce,
    label_position_mp_correction_setrank,
)
from vlm_multilabel.losses.label_positions import extract_label_probabilities, find_label_positions


def test_find_label_positions_returns_first_response_label():
    labels = torch.tensor([[-100, 7, 8], [-100, 8, 7]])
    token_ids = torch.tensor([7, 8])
    positions, valid = find_label_positions(labels, token_ids)
    assert positions.tolist() == [0, 0]
    assert valid.tolist() == [True, True]


def test_find_label_positions_marks_missing_labels_invalid():
    labels = torch.tensor([[-100, -100]])
    positions, valid = find_label_positions(labels, torch.tensor([7, 8]))
    assert positions.tolist() == [0]
    assert valid.tolist() == [False]


def test_extract_label_probabilities_uses_float32():
    logits = torch.tensor([[[100.0], [0.0]]], dtype=torch.float16)
    labels = torch.tensor([[-100, 0]])
    probabilities, valid = extract_label_probabilities(logits, labels, torch.tensor([0]))
    assert probabilities.dtype == torch.float32
    assert probabilities.item() == pytest.approx(1.0, abs=1e-6)
    assert valid.item() is True


def test_bce_ignores_invalid_samples():
    logits = torch.zeros(2, 3, 1)
    labels = torch.tensor([[-100, 0, -100], [-100, -100, -100]])
    label_vector = torch.tensor([[1.0], [0.0]])
    loss = label_position_bce(logits, labels, label_vector, torch.tensor([0]))
    assert loss.item() == pytest.approx(0.6931471805599453)


def test_mp_correction_is_zero_for_single_positive_sample():
    logits = torch.randn(1, 3, 2)
    labels = torch.tensor([[-100, 0, -100]])
    label_vector = torch.tensor([[1.0, 0.0]])
    loss = label_position_mp_correction_setrank(
        logits, labels, label_vector, torch.tensor([0, 1]), 1
    )
    assert loss.item() == 0.0


def test_mp_correction_pushes_valid_set_above_invalid_set():
    logits = torch.zeros(1, 3, 3)
    logits[0, 0] = torch.tensor([5.0, -5.0, -5.0])
    labels = torch.tensor([[-100, 0, -100]])
    label_vector = torch.tensor([[1.0, 1.0, 0.0]])
    loss = label_position_mp_correction_setrank(
        logits, labels, label_vector, torch.tensor([0, 1, 2]), 1
    )
    assert loss.item() == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(MP_MARGIN - 10.0))), abs=1e-6)
