"""Tests for the aux-head control baseline (pure-tensor paths)."""
import torch

from vlm_multilabel.losses.aux_head import (
    AuxHead,
    aux_head_bce_loss,
    gather_last_hidden,
)


def test_gather_last_hidden_right_padding():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    gathered = gather_last_hidden(hidden, mask)
    assert torch.equal(gathered[0], hidden[0, 2])
    assert torch.equal(gathered[1], hidden[1, 1])


def test_gather_last_hidden_left_padding():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    gathered = gather_last_hidden(hidden, mask)
    assert torch.equal(gathered[0], hidden[0, 2])
    assert torch.equal(gathered[1], hidden[1, 2])


def test_aux_head_bce_loss_matches_manual():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = aux_head_bce_loss(logits, targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    )
    assert torch.allclose(loss, expected)


def test_aux_head_save_load_roundtrip(tmp_path):
    head = AuxHead(hidden_size=4, num_classes=3)
    path = tmp_path / "aux_head.pt"
    head.save(str(path))
    restored = AuxHead.load(str(path))
    assert restored.hidden_size == 4
    assert restored.num_classes == 3
    sample = torch.randn(2, 4)
    assert torch.allclose(head(sample), restored(sample))


def test_gather_rejects_all_zero_mask_row():
    hidden = torch.zeros(1, 2, 2)
    mask = torch.zeros(1, 2, dtype=torch.long)
    gathered = gather_last_hidden(hidden, mask)
    assert gathered.shape == (1, 2)
