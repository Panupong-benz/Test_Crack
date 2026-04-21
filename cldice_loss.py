"""clDice loss for tubular structure segmentation (Shit et al., CVPR 2021).

clDice = centerline Dice. Designed for thin connected structures
(vessels / roads / cracks) where pixel-level Dice misses topology errors:
losing 5 pixels along a 200-pixel crack barely moves Dice but breaks
connectivity. clDice penalises broken or shifted skeletons directly.

Soft skeletonisation via iterative morphological thinning (min/max pooling
through F.max_pool2d) makes the operation differentiable end-to-end.

Reference:
    Shit, S. et al. (2021). "clDice — a Novel Topology-Preserving Loss
    Function for Tubular Structure Segmentation". CVPR.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sam3.train.loss.loss_fns import Masks


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable skeletonisation. Input/output: [N, 1, H, W] in [0, 1]."""
    img1 = _soft_open(x)
    skel = F.relu(x - img1)
    for _ in range(iters):
        x = _soft_erode(x)
        img1 = _soft_open(x)
        delta = F.relu(x - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def _maybe_downsample(x: torch.Tensor, max_size: int) -> torch.Tensor:
    """Downsample [N, 1, H, W] so max(H, W) <= max_size. Skip if already small enough
    or max_size <= 0 (disabled)."""
    if max_size <= 0:
        return x
    h, w = x.shape[-2:]
    m = max(h, w)
    if m <= max_size:
        return x
    scale = max_size / float(m)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def soft_cldice(
    pred_logits_2d: torch.Tensor,
    target_2d: torch.Tensor,
    num_boxes: float,
    iters: int = 3,
    smooth: float = 1.0,
    max_size: int = 384,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Per-instance clDice summed across batch and divided by num_boxes
    to match the convention of sam3's dice_loss / sigmoid_focal_loss.

    Memory notes:
      - `max_size` downsamples pred/target before skeletonisation. clDice is a
        topological loss, so modest downsampling still captures connectivity.
        Set <=0 to disable.
      - Target skeletonisation runs under no_grad (target has no grad anyway).
      - `use_checkpoint` wraps pred skeletonisation in torch.utils.checkpoint
        to trade compute for memory.
    """
    if pred_logits_2d.numel() == 0:
        return pred_logits_2d.sum() * 0.0

    pred = torch.sigmoid(pred_logits_2d.float()).unsqueeze(1)  # [N, 1, H, W]
    target = target_2d.float().unsqueeze(1)

    pred = _maybe_downsample(pred, max_size)
    target = _maybe_downsample(target, max_size)

    if use_checkpoint and pred.requires_grad:
        from torch.utils.checkpoint import checkpoint
        skel_pred = checkpoint(soft_skeletonize, pred, iters, use_reentrant=False)
    else:
        skel_pred = soft_skeletonize(pred, iters)

    with torch.no_grad():
        skel_target = soft_skeletonize(target, iters)

    # Topology precision: GT skeleton inside predicted volume
    tprec_num = (skel_pred * target).sum(dim=(1, 2, 3))
    tprec = (tprec_num + smooth) / (skel_pred.sum(dim=(1, 2, 3)) + smooth)

    # Topology sensitivity: predicted skeleton inside GT volume
    tsens_num = (skel_target * pred).sum(dim=(1, 2, 3))
    tsens = (tsens_num + smooth) / (skel_target.sum(dim=(1, 2, 3)) + smooth)

    cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    per_instance = 1.0 - cldice
    return per_instance.sum() / max(float(num_boxes), 1.0)


class MasksWithCLDice(Masks):
    """sam3 Masks loss + clDice term.

    Adds key 'loss_cldice' to the returned losses dict. Include it in
    weight_dict to control its contribution, e.g.:

        MasksWithCLDice(
            weight_dict={
                "loss_mask":   200.0,
                "loss_dice":    50.0,
                "loss_cldice":  30.0,
            },
            focal_alpha=0.85,
            focal_gamma=3.0,
            cldice_iters=3,
        )

    clDice is computed on the matched, upsampled 2D masks. When
    `num_sample_points` is set (point-sampled training path) clDice is
    skipped because skeletonisation requires the full 2D mask.
    """

    def __init__(
        self,
        *args,
        cldice_iters: int = 3,
        cldice_smooth: float = 1.0,
        cldice_max_size: int = 384,
        cldice_use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cldice_iters = cldice_iters
        self.cldice_smooth = cldice_smooth
        self.cldice_max_size = cldice_max_size
        self.cldice_use_checkpoint = cldice_use_checkpoint

    def get_loss(self, outputs, targets, indices, num_boxes):
        losses = super().get_loss(outputs, targets, indices, num_boxes)
        device = losses["loss_mask"].device

        if "loss_cldice" not in self.weight_dict:
            losses["loss_cldice"] = torch.tensor(0.0, device=device)
            return losses
        if targets["masks"] is None or self.num_sample_points is not None:
            losses["loss_cldice"] = torch.tensor(0.0, device=device)
            return losses

        # Re-derive the matched 2D masks (parent flattens before computing loss).
        src_masks = outputs["pred_masks"]
        target_masks = (
            targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
        ).to(src_masks)
        keep = (
            targets["is_valid_mask"]
            if indices[2] is None
            else targets["is_valid_mask"][indices[2]]
        )
        src_masks = src_masks[(indices[0], indices[1])][keep]
        target_masks = target_masks[keep]

        if src_masks.numel() == 0 or target_masks.numel() == 0:
            losses["loss_cldice"] = torch.tensor(0.0, device=device)
            return losses

        if len(src_masks.shape) == 3:
            src_masks = src_masks[:, None]
        if src_masks.dtype == torch.bfloat16:
            src_masks = src_masks.to(dtype=torch.float32)
        src_masks = F.interpolate(
            src_masks, size=target_masks.shape[-2:],
            mode="bilinear", align_corners=False,
        )[:, 0]  # [N, H, W]

        losses["loss_cldice"] = soft_cldice(
            src_masks, target_masks, num_boxes,
            iters=self.cldice_iters,
            smooth=self.cldice_smooth,
            max_size=self.cldice_max_size,
            use_checkpoint=self.cldice_use_checkpoint,
        )
        return losses
