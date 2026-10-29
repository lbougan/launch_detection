import torch

from services.training.losses import CombinedLoss


def test_ignore_pixels_do_not_contribute():
    loss_fn = CombinedLoss()
    logits = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    weight = torch.ones(1, 1, 4, 4)
    base = loss_fn(logits, target, weight).item()
    # Make one pixel "very wrong" but ignored: loss must not change.
    logits2 = logits.clone()
    logits2[0, 0, 0, 0] = 10.0
    weight2 = weight.clone()
    weight2[0, 0, 0, 0] = 0.0
    assert abs(loss_fn(logits2, target, weight2).item() - base) < 1e-6
    # And not ignored: loss must increase.
    assert loss_fn(logits2, target, weight).item() > base


def test_no_label_smoothing_targets_reach_zero_loss():
    loss_fn = CombinedLoss()
    logits = torch.full((1, 1, 2, 2), 20.0)
    target = torch.ones(1, 1, 2, 2)
    assert loss_fn(logits, target, torch.ones_like(target)).item() < 1e-3


def test_ignore_pixels_do_not_contribute_to_dice():
    loss_fn = CombinedLoss()
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2, :2] = 1.0
    logits = torch.where(target > 0, torch.tensor(5.0), torch.tensor(-5.0))
    ignored = torch.ones_like(target)
    ignored[0, 0, 3, 3] = 0.0
    # Baseline: the pixel is ignored (weight 0) and predicted correctly.
    base = loss_fn(logits, target, ignored).item()
    # Same pixel, ignored, but predicted strongly positive: loss unchanged.
    wrong = logits.clone()
    wrong[0, 0, 3, 3] = 10.0
    assert abs(loss_fn(wrong, target, ignored).item() - base) < 1e-6
    # Not ignored: loss strictly greater.
    assert loss_fn(wrong, target, torch.ones_like(target)).item() > base
