import numpy as np
import pytest

from vlm_multilabel.metrics import compute_accuracy, compute_auc, compute_map, compute_threshold_sweep


def test_compute_map_skips_empty_classes():
    y_true = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.float32)
    y_pred = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.2]], dtype=np.float32)
    mean_ap, per_class = compute_map(y_true, y_pred)
    assert len(per_class) == 3
    assert np.isnan(per_class[1])
    valid = [v for v in per_class if not np.isnan(v)]
    assert mean_ap == pytest.approx(sum(valid) / len(valid))


def test_compute_accuracy_returns_nan_for_constant_classes():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.9, 0.1], [0.2, 0.2]], dtype=np.float32)
    mean_accuracy, per_class = compute_accuracy(y_true, y_pred)
    assert per_class[0] == pytest.approx(1.0)
    assert np.isnan(per_class[1])
    assert mean_accuracy == pytest.approx(1.0)


def test_compute_auc_skips_constant_classes():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.9, 0.1], [0.2, 0.2]], dtype=np.float32)
    mean_auc, per_class = compute_auc(y_true, y_pred)
    assert per_class[0] == pytest.approx(1.0)
    assert np.isnan(per_class[1])


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        compute_map(np.zeros((2, 2)), np.zeros((3, 2)))


def test_compute_threshold_sweep_returns_fixed_grid():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[0.9, 0.001], [0.001, 0.9]], dtype=np.float32)
    sweep = compute_threshold_sweep(y_true, y_pred)
    assert [row["threshold"] for row in sweep] == [
        0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
        0.70, 0.80, 0.90, 0.95, 0.99,
    ]
    assert sweep[0]["mean_balanced_accuracy"] == pytest.approx(1.0)

from vlm_multilabel.metrics import compute_brier, compute_ece, compute_nll


def test_compute_ece_uses_equal_mass_bins_and_macro_average():
    y_true = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
    y_pred = np.array(
        [
            [0.1, 0.0],
            [0.3, 0.0],
            [0.7, 1.0],
            [0.9, 1.0],
        ],
        dtype=np.float32,
    )
    ece, per_label = compute_ece(y_true, y_pred, num_bins=2)
    # For class 0, the two equal-mass bins contribute 0.5 * |0 - 0.2| each.
    assert ece == pytest.approx(0.1)
    assert per_label == pytest.approx([0.2, 0.0])


def test_compute_ece_skips_constant_label_columns():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.2], [0.3, 0.8]], dtype=np.float32)
    ece, per_label = compute_ece(y_true, y_pred, num_bins=2)
    assert len(per_label) == 1
    assert ece == pytest.approx(per_label[0])


def test_compute_ece_is_zero_for_perfectly_calibrated_binary_outputs():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = y_true.copy()
    ece, per_label = compute_ece(y_true, y_pred, num_bins=2)
    assert ece == pytest.approx(0.0)
    assert per_label == pytest.approx([0.0, 0.0])


def test_compute_brier_matches_mean_squared_error_and_macro_average():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    brier, per_label = compute_brier(y_true, y_pred)
    expected_label_0 = ((1 - 0.8) ** 2 + (0 - 0.3) ** 2) / 2
    expected_label_1 = ((0 - 0.2) ** 2 + (1 - 0.7) ** 2) / 2
    assert per_label == pytest.approx([expected_label_0, expected_label_1])
    assert brier == pytest.approx((expected_label_0 + expected_label_1) / 2)


def test_compute_brier_skips_constant_label_columns():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.2], [0.3, 0.8]], dtype=np.float32)
    brier, per_label = compute_brier(y_true, y_pred)
    assert len(per_label) == 1
    assert brier == pytest.approx(per_label[0])


def test_compute_brier_returns_zero_when_all_labels_are_constant():
    y_true = np.zeros((2, 2), dtype=np.float32)
    y_pred = np.full((2, 2), 0.25, dtype=np.float32)
    assert compute_brier(y_true, y_pred) == (0.0, [])


def test_compute_nll_matches_binary_cross_entropy_macro_average():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.full((2, 2), 0.5, dtype=np.float32)
    nll, per_label = compute_nll(y_true, y_pred)
    assert per_label == pytest.approx([np.log(2.0), np.log(2.0)])
    assert nll == pytest.approx(np.log(2.0))


def test_compute_nll_clips_extreme_probabilities():
    # Include both positive and negative targets so the label is not skipped.
    # Both predictions are maximally wrong, so clipping must keep NLL finite.
    y_true = np.array([[1], [0]], dtype=np.float32)
    y_pred = np.array([[0.0], [1.0]], dtype=np.float32)
    nll, per_label = compute_nll(y_true, y_pred, eps=1e-7)
    expected = -np.log(1e-7)
    assert per_label == pytest.approx([expected])
    assert nll == pytest.approx(expected)


def test_compute_nll_skips_constant_label_columns():
    y_true = np.array([[1, 0], [0, 0]], dtype=np.float32)
    y_pred = np.full((2, 2), 0.5, dtype=np.float32)
    nll, per_label = compute_nll(y_true, y_pred)
    assert len(per_label) == 1
    assert nll == pytest.approx(np.log(2.0))


def test_compute_nll_returns_zero_when_all_labels_are_constant():
    y_true = np.zeros((2, 2), dtype=np.float32)
    y_pred = np.full((2, 2), 0.5, dtype=np.float32)
    assert compute_nll(y_true, y_pred) == (0.0, [])


@pytest.mark.parametrize(
    "metric",
    [compute_ece, compute_brier, compute_nll],
)
def test_calibration_metrics_validate_shapes(metric):
    with pytest.raises(ValueError):
        metric(np.zeros((2, 2)), np.zeros((3, 2)))
