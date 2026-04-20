#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Phase-1 inference improvements (thesis progress report §6):
    * Detection threshold lowered to 0.30 (was 0.5)
    * NMS IoU threshold lowered to 0.30 (was 0.5)
    * CLAHE contrast enhancement before inference (LAB space, clipLimit=3.0)
    * Optional sliding-window inference on 640 px tiles with 25 % overlap
      (CLAHE applied once to full image; cross-tile NMS dedupes boxes)
    * Optional morphological post-processing (§4.3): MORPH_CLOSE with line
      kernels to bridge crack gaps, MORPH_OPEN to strip noise
    * Optional skeletonization for centerline / length measurement (§4.3)
    * Seeded for bit-reproducible runs; tqdm progress bar over tiles

Usage:
    # Text prompt inference (defaults: thresh=0.3, nms=0.3, CLAHE on)
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Sliding-window with post-processing + skeleton on a large wall image
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --sliding-window \
        --tile-size 640 --tile-overlap 0.25 \
        --postprocess --pp-close 20 --pp-open 3 \
        --skeletonize \
        --output output.png
"""

import argparse
import os
import random
from typing import List, Optional, Union

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

try:
    import cv2
except ImportError as _e:
    raise ImportError(
        "cv2 (opencv-python) is required for CLAHE preprocessing. "
        "Install with: pip install opencv-python"
    ) from _e

# Progress bar (graceful fallback if tqdm not installed)
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **_kw):
        return iterable

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


class SAM3LoRAInference:
    """SAM3 model with LoRA for inference."""

    def __init__(
        self,
        config_path: str,
        weights_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.30,
        nms_iou_threshold: float = 0.30,
        use_clahe: bool = True,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid: int = 8,
        use_postprocess: bool = False,
        pp_close_kernel: int = 20,
        pp_open_kernel: int = 3,
        use_skeleton: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize SAM3 with LoRA.

        Args:
            config_path: Path to training config YAML
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.30,
                lowered from 0.5 per thesis report §6 to improve recall on thin cracks)
            nms_iou_threshold: IoU threshold for NMS (default: 0.30, lowered from 0.5
                so closely-spaced crack segments can co-exist)
            use_clahe: Apply CLAHE contrast enhancement before inference (default: True).
                Operates in LAB color space so marker-pen hue is preserved.
            clahe_clip_limit: CLAHE clip limit (default: 3.0, per thesis report)
            clahe_tile_grid: CLAHE tile grid size NxN (default: 8)
            use_postprocess: Apply morphological post-processing to masks
                (default: False — opt-in). Close with (close_k, 1) and
                (1, close_k) kernels to bridge crack gaps, then open with
                (open_k, open_k) to remove isolated noise. Per thesis report §4.3.
            pp_close_kernel: Line-kernel length for gap bridging (default: 20)
            pp_open_kernel: Square-kernel size for noise removal (default: 3)
            use_skeleton: Also return a 1-px skeleton (centerline) of the mask
                in results[q]['skeleton']. Useful for length / orientation
                measurement downstream (default: False). Per thesis report §4.3.
            device: Device to run on (default: "cuda")
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Auto-detect weights if not provided
        if weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
            print(f"ℹ️  Auto-detected weights: {weights_path}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.use_postprocess = use_postprocess
        self.pp_close_kernel = pp_close_kernel
        self.pp_open_kernel = pp_open_kernel
        self.use_skeleton = use_skeleton
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"🔧 Initializing SAM3 + LoRA...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")
        print(f"   CLAHE preprocessing: {'ON' if use_clahe else 'OFF'}"
              + (f" (clip={clahe_clip_limit}, grid={clahe_tile_grid}x{clahe_tile_grid})"
                 if use_clahe else ""))
        print(f"   Mask post-processing: {'ON' if use_postprocess else 'OFF'}"
              + (f" (close={pp_close_kernel}, open={pp_open_kernel})"
                 if use_postprocess else ""))
        print(f"   Skeletonization: {'ON' if use_skeleton else 'OFF'}")

        # Build base model
        print("\n📦 Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        print(f"💾 Loading LoRA weights from {weights_path}...")
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        print("✅ SAM3 + LoRA ready for inference!\n")

    # ------------------------------------------------------------------
    # Preprocessing helpers (thesis report §4.2 / §6)
    # ------------------------------------------------------------------
    def _apply_clahe(self, pil_image: PILImage.Image) -> PILImage.Image:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB
        color space. Enhances local contrast of crack edges against concrete
        without over-exposing background, and preserves the hue of blue/red
        marker-pen strokes (they're encoded in a, b channels — we only
        equalize L).
        """
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_grid, self.clahe_tile_grid),
        )
        l_eq = clahe.apply(l)
        merged = cv2.merge([l_eq, a, b])
        out_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(out_rgb)

    def _preprocess(self, pil_image: PILImage.Image) -> PILImage.Image:
        """Apply all enabled preprocessing steps and return a new PIL image."""
        if self.use_clahe:
            pil_image = self._apply_clahe(pil_image)
        return pil_image

    # ------------------------------------------------------------------
    # Post-processing helpers (thesis report §4.3)
    # ------------------------------------------------------------------
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological post-processing to a binary mask:
          * MORPH_CLOSE with (close_k, 1) + (1, close_k) line kernels →
            bridges small gaps along crack paths the model detected
            as separate segments.
          * MORPH_OPEN with (open_k, open_k) square kernel →
            removes isolated noise blobs that aren't part of connected
            crack structures.
        Input: HxW bool/uint8 array. Returns: HxW bool array.
        """
        m = (mask > 0).astype(np.uint8)
        k_h = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.pp_close_kernel, 1)
        )
        k_v = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, self.pp_close_kernel)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_h)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_v)
        if self.pp_open_kernel > 0:
            k_o = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.pp_open_kernel, self.pp_open_kernel)
            )
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_o)
        return m.astype(bool)

    def _skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Reduce a binary mask to its 1-pixel centerline using skimage's
        `skeletonize`. Useful for length / orientation measurement.
        Input: HxW bool/uint8 (or (N,H,W) stack → per-instance skeleton OR'd).
        Returns: HxW bool.
        """
        from skimage.morphology import skeletonize
        if mask.ndim == 3:
            mask = mask.any(axis=0)
        return skeletonize(mask > 0).astype(bool)

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        """
        Create a SAM3 datapoint from image and text prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, "PILImage.Image"],
        text_prompts: List[str],
        apply_preprocess: bool = True,
        apply_postprocess: Optional[bool] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run inference on an image with text prompts.

        Args:
            image_path: Path to input image, OR a PIL.Image.Image. Accepting
                a PIL image lets callers (e.g. predict_sliding_window) avoid
                a disk round-trip per tile.
            text_prompts: List of text queries (e.g., ["crack", "defect"])
            apply_preprocess: Whether to run CLAHE / any preprocessing step
                on the loaded image. Default True. Set to False when the
                caller has already preprocessed the image (e.g. sliding-
                window inference applies CLAHE once to the full image and
                crops tiles from the preprocessed canvas — each tile then
                passes through predict() with apply_preprocess=False to
                avoid a second, tile-local CLAHE pass that would produce
                tile-boundary contrast artifacts).
            apply_postprocess: Override self.use_postprocess for this call.
                None (default) → use the class-level setting. Set to False
                from predict_sliding_window so post-processing can be done
                ONCE on the final union mask (better gap bridging across
                tile boundaries than per-tile postprocess).
            verbose: If False, suppress per-call prints (used internally by
                sliding-window inference to keep the tqdm progress bar
                readable).

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...],
                    'skeleton': [...] (optional)},
                ...
            }
        """
        # Load image — accept either a path or a PIL.Image
        if isinstance(image_path, PILImage.Image):
            pil_image_raw = image_path.convert("RGB")
            src_repr = "<PIL.Image>"
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_image_raw = PILImage.open(image_path).convert("RGB")
            src_repr = image_path

        if verbose:
            print(f"📷 Loaded image: {src_repr}")
            print(f"   Size: {pil_image_raw.size}")
            print(f"   Prompts: {text_prompts}")

        # Apply CLAHE / other preprocessing (no-op if disabled or if caller
        # already preprocessed upstream)
        if apply_preprocess:
            pil_image = self._preprocess(pil_image_raw)
            if verbose and self.use_clahe:
                print("   Preprocessing: CLAHE applied (LAB space, L-channel only)")
        else:
            pil_image = pil_image_raw
            if verbose:
                print("   Preprocessing: skipped (already applied by caller)")

        # Resolve post-processing toggle
        do_pp = self.use_postprocess if apply_postprocess is None else apply_postprocess

        if verbose:
            print("\n🔮 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt
            datapoint = self.create_datapoint(pil_image, [prompt])

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and convert from cxcywh to xyxy
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # Apply NMS to remove overlapping boxes
                keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
                boxes_xyxy = boxes_xyxy[keep_nms]
                kept_scores = kept_scores[keep_nms]
                num_keep = len(keep_nms)

                # Get masks and resize to original size
                if pred_masks is not None:
                    # Apply NMS filtering to masks too
                    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                    # Resize masks to original image size
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        masks_small.unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                # ---- Optional morphological post-processing (per instance) ----
                if do_pp and masks_np is not None and len(masks_np) > 0:
                    masks_np = np.stack(
                        [self._postprocess_mask(m) for m in masks_np], axis=0
                    )

                entry = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'num_detections': num_keep,
                }

                # ---- Optional skeleton (1-px centerline of union mask) ----
                if self.use_skeleton and masks_np is not None:
                    entry['skeleton'] = self._skeletonize_mask(masks_np)

                results[query_idx] = entry
                if verbose:
                    print(f"   '{prompt}': {num_keep} detections after NMS "
                          f"(max score: {kept_scores.max().item():.3f})"
                          + (" [postprocessed]" if do_pp else ""))
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'num_detections': 0
                }
                if verbose:
                    print(f"   '{prompt}': 0 detections")

        # Store ORIGINAL (pre-CLAHE) image for visualization so users see their
        # untouched photo with masks overlaid. The CLAHE-enhanced version is only
        # fed to the model.
        results['_image'] = pil_image_raw
        results['_image_preprocessed'] = pil_image

        return results

    # ------------------------------------------------------------------
    # Sliding-window inference (thesis report §6 + §7 phase-3 item 12)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_sliding_window(
        self,
        image_path: str,
        text_prompts: List[str],
        tile_size: int = 640,
        overlap: float = 0.25,
        show_progress: bool = True,
    ) -> dict:
        """
        Run inference on overlapping tiles of the full-resolution image, then
        stitch the per-tile masks back into full-image coordinates. Useful for
        high-res wall photos where a single 1008×1008 forward pass loses
        hairline cracks to downsampling.

        Pipeline (fixed vs. earlier version):
          1. CLAHE is applied ONCE on the full image (not per-tile) — avoids
             tile-boundary contrast artifacts from local histograms.
          2. Each tile is cropped from the preprocessed canvas in-memory
             (no temp-file disk round-trip) and passed as a PIL image to
             predict() with apply_preprocess=False and apply_postprocess=False.
          3. Masks from different tiles are merged into a logical-OR union
             in full-image coordinates.
          4. Boxes/scores are concatenated across tiles, then DEDUPED with
             cross-tile NMS so a crack seen by multiple overlapping tiles
             isn't counted N times.
          5. Morphological post-processing (if enabled) is run ONCE on the
             final union mask — this lets MORPH_CLOSE bridge crack gaps
             that straddle tile boundaries.
          6. Skeletonization (if enabled) runs on the final post-processed
             union mask.

        Args:
            image_path: Path to input image OR a PIL.Image.Image
            text_prompts: List of text queries
            tile_size: Tile side length in pixels (default: 640)
            overlap: Fractional overlap between adjacent tiles [0, 1)
                (default: 0.25 → stride = 480 for tile_size=640)
            show_progress: Show a tqdm progress bar over tiles (default: True).
                Set False to silence (e.g. for CI logs).

        Returns:
            dict with the same schema as predict(), but masks/boxes/scores
            are aggregated across all tiles and remapped to full-image coords.
        """
        assert 0.0 <= overlap < 1.0, "overlap must be in [0, 1)"

        # Accept either a path or a PIL image
        if isinstance(image_path, PILImage.Image):
            pil_full = image_path.convert("RGB")
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_full = PILImage.open(image_path).convert("RGB")
        W, H = pil_full.size
        stride = max(1, int(round(tile_size * (1.0 - overlap))))

        print(f"📷 Sliding-window inference: {W}×{H} image, "
              f"tile={tile_size}, overlap={overlap:.0%}, stride={stride}")

        # ---- Preprocess the FULL image ONCE (fix for CLAHE tile-boundary issue) ----
        # We apply CLAHE on the full-resolution canvas so every tile sees a
        # globally-consistent contrast. Per-tile CLAHE would compute local
        # histograms against a 640×640 context instead of the real image
        # context, creating contrast jumps at tile boundaries.
        if self.use_clahe:
            pil_full_proc = self._preprocess(pil_full)
            print("   Preprocessing: CLAHE applied ONCE to full image "
                  "(tiles crop from the preprocessed canvas)")
        else:
            pil_full_proc = pil_full

        # Degenerate case: image smaller than one tile → fall back to regular predict
        if W <= tile_size and H <= tile_size:
            print("   Image ≤ tile_size, falling back to single-image predict()")
            return self.predict(image_path, text_prompts)

        # Build tile origins; always include a final tile flush with the edge
        def _origins(length: int) -> List[int]:
            if length <= tile_size:
                return [0]
            xs = list(range(0, length - tile_size + 1, stride))
            if xs[-1] + tile_size < length:
                xs.append(length - tile_size)
            return xs

        x_origins = _origins(W)
        y_origins = _origins(H)
        n_tiles = len(x_origins) * len(y_origins)
        print(f"   Tiles: {len(x_origins)}×{len(y_origins)} = {n_tiles}")

        # Per-prompt accumulators
        merged = {idx: {
            "prompt": text_prompts[idx],
            "mask_union": np.zeros((H, W), dtype=bool),
            "boxes": [],
            "scores": [],
        } for idx in range(len(text_prompts))}

        # Flatten tile positions so tqdm can show a single progress bar
        tile_positions = [(xo, yo) for yo in y_origins for xo in x_origins]

        iterator = _tqdm(tile_positions, desc="Tiles", unit="tile",
                         disable=not show_progress)

        for tile_idx, (xo, yo) in enumerate(iterator, start=1):
            # Crop tile directly from the PREPROCESSED full canvas —
            # no temp-file round-trip (passes PIL straight to predict())
            tile = pil_full_proc.crop((xo, yo, xo + tile_size, yo + tile_size))

            # apply_preprocess=False  → pil_full_proc is already CLAHE'd
            # apply_postprocess=False → post-processing is done ONCE on
            #                           the final union mask (better gap
            #                           bridging across tile boundaries)
            # verbose=False           → keeps tqdm readable
            tile_results = self.predict(
                tile,
                text_prompts,
                apply_preprocess=False,
                apply_postprocess=False,
                verbose=False,
            )

            for q_idx in range(len(text_prompts)):
                r = tile_results.get(q_idx, {})
                if not r or r.get("num_detections", 0) == 0:
                    continue

                # Shift boxes from tile coords → full-image coords
                boxes = r["boxes"].copy()  # (N, 4) xyxy
                boxes[:, [0, 2]] += xo
                boxes[:, [1, 3]] += yo
                merged[q_idx]["boxes"].append(boxes)
                merged[q_idx]["scores"].append(r["scores"])

                # OR-merge masks into the full-image canvas
                if r.get("masks") is not None:
                    tile_masks = r["masks"]  # (N, tile_h, tile_w) bool
                    any_tile = tile_masks.any(axis=0)
                    # Guard against rare shape mismatches
                    th, tw = any_tile.shape
                    merged[q_idx]["mask_union"][
                        yo:yo + th, xo:xo + tw
                    ] |= any_tile

            # Memory hygiene — prevents VRAM creep across many tiles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ---- Finalize per-prompt aggregates (with cross-tile NMS) ----
        final = {}
        for q_idx, agg in merged.items():
            if len(agg["boxes"]) == 0:
                final[q_idx] = {
                    "prompt": agg["prompt"],
                    "boxes": None,
                    "scores": None,
                    "masks": None,
                    "num_detections": 0,
                }
                continue

            all_boxes = np.concatenate(agg["boxes"], axis=0)    # (N, 4) xyxy
            all_scores = np.concatenate(agg["scores"], axis=0)  # (N,)
            n_raw = int(all_boxes.shape[0])

            # ---- Cross-tile NMS: dedupe boxes/scores from overlapping tiles ----
            # Without this, a crack sitting in a tile overlap region is
            # counted once per tile that saw it, inflating num_detections
            # and skewing scores.max(). The OR-merged mask is unaffected.
            if n_raw > 1:
                boxes_t = torch.from_numpy(all_boxes).float()
                scores_t = torch.from_numpy(all_scores).float()
                keep_idx = nms(boxes_t, scores_t, self.nms_iou_threshold).cpu().numpy()
                all_boxes = all_boxes[keep_idx]
                all_scores = all_scores[keep_idx]
            n_kept = int(all_boxes.shape[0])

            # ---- Post-process the full union mask (once, globally) ----
            # Doing it here (not per-tile) lets MORPH_CLOSE bridge crack
            # gaps that straddle tile boundaries.
            union_raw = agg["mask_union"]
            if self.use_postprocess:
                union_pp = self._postprocess_mask(union_raw)
            else:
                union_pp = union_raw

            # mask_union is a single merged binary mask; expose as (1, H, W)
            # to keep downstream code (which does masks.any(axis=0)) happy.
            union_mask = union_pp[None, :, :]

            entry = {
                "prompt": agg["prompt"],
                "boxes": all_boxes,
                "scores": all_scores,
                "masks": union_mask,
                "num_detections": n_kept,
            }

            # Optional skeleton on the final (post-processed) union mask
            if self.use_skeleton:
                entry["skeleton"] = self._skeletonize_mask(union_pp)

            final[q_idx] = entry

            print(f"\n   ✓ '{agg['prompt']}': "
                  f"{n_kept} detections after cross-tile NMS "
                  f"(from {n_raw} raw across tiles, "
                  f"union mask: {int(union_mask.sum())} pixels"
                  + (", postprocessed" if self.use_postprocess else "") + ")")

        final["_image"] = pil_full
        # Expose the full-image preprocessed canvas for --save-preprocessed.
        # This NOW matches exactly what every tile saw (we cropped from it).
        final["_image_preprocessed"] = pil_full_proc
        return final

    # ------------------------------------------------------------------
    # Test-time augmentation (Tier 1, item 2 in next-steps roadmap)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_tta(
        self,
        image_path: Union[str, "PILImage.Image"],
        text_prompts: List[str],
        tta_transforms: tuple = ("identity", "hflip", "vflip", "rot180"),
        aggregate: str = "mean",
        threshold: float = 0.5,
        min_component_area: int = 5,
        apply_preprocess: bool = True,
        apply_postprocess: Optional[bool] = None,
        verbose: bool = True,
    ) -> dict:
        """Test-time augmentation via geometric flip / rotation ensemble.

        Runs predict() N times on transformed copies of the image, inverse-
        transforms each pass's union mask back to the original orientation,
        then aggregates per-pixel and re-derives instances by connected
        components. For binary semantic-style targets like cracks, mask-level
        averaging is more principled than per-instance NMS reconciliation
        across passes (instance identity is unstable under rotation).

        Supported transforms: identity, hflip, vflip, rot180.
        rot90 / rot270 are intentionally omitted from the v1 default because
        they swap H↔W; safe but adds bookkeeping. Add manually if desired.

        Args:
            tta_transforms: subset of {"identity","hflip","vflip","rot180"}.
            aggregate:      "mean" (probabilistic), "max" (union), or "vote"
                            (per-pixel majority across passes).
            threshold:      Binarisation threshold on the aggregated map.
            min_component_area: Drop connected components smaller than this
                            (after aggregation) to suppress speckle.
        """
        if isinstance(image_path, PILImage.Image):
            pil_raw = image_path.convert("RGB")
            src_repr = "<PIL.Image>"
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_raw = PILImage.open(image_path).convert("RGB")
            src_repr = image_path

        orig_w, orig_h = pil_raw.size
        do_pp = self.use_postprocess if apply_postprocess is None else apply_postprocess

        # forward (PIL → PIL) and inverse (np mask → np mask) for each op
        TTA_OPS = {
            "identity": (lambda im: im,
                         lambda m: m),
            "hflip":    (lambda im: im.transpose(PILImage.FLIP_LEFT_RIGHT),
                         lambda m: np.ascontiguousarray(m[..., :, ::-1])),
            "vflip":    (lambda im: im.transpose(PILImage.FLIP_TOP_BOTTOM),
                         lambda m: np.ascontiguousarray(m[..., ::-1, :])),
            "rot180":   (lambda im: im.transpose(PILImage.ROTATE_180),
                         lambda m: np.ascontiguousarray(np.rot90(m, k=2, axes=(-2, -1)))),
        }
        for name in tta_transforms:
            if name not in TTA_OPS:
                raise ValueError(f"Unknown TTA transform: {name}. "
                                 f"Supported: {list(TTA_OPS)}")
        if aggregate not in ("mean", "max", "vote"):
            raise ValueError(f"Unknown aggregate: {aggregate}")

        if verbose:
            print(f"📷 Loaded image: {src_repr}")
            print(f"   Size: {orig_w}x{orig_h}")
            print(f"   TTA passes: {len(tta_transforms)} ({list(tta_transforms)})")
            print(f"   Aggregate: {aggregate}, threshold: {threshold}")

        # Per-prompt accumulators of union-masks at original orientation.
        # Shape per slot: float32 [orig_h, orig_w]
        accum: dict = {}

        for tta_name in tta_transforms:
            fwd, inv = TTA_OPS[tta_name]
            pil_t = fwd(pil_raw)

            pass_result = self.predict(
                pil_t,
                text_prompts,
                apply_preprocess=apply_preprocess,
                apply_postprocess=False,  # do post-process ONCE on the union
                verbose=False,
            )

            for q_idx, entry in pass_result.items():
                if not isinstance(q_idx, int):
                    continue
                masks = entry.get("masks")
                if masks is None or len(masks) == 0:
                    union_t = np.zeros((pil_t.size[1], pil_t.size[0]), dtype=np.float32)
                else:
                    union_t = (masks.sum(axis=0) > 0).astype(np.float32)

                union_orig = inv(union_t)
                if union_orig.shape != (orig_h, orig_w):
                    raise RuntimeError(
                        f"TTA inverse shape mismatch: got {union_orig.shape}, "
                        f"expected {(orig_h, orig_w)}"
                    )

                slot = accum.setdefault(q_idx, {
                    "prompt":   entry.get("prompt"),
                    "mask_sum": np.zeros((orig_h, orig_w), dtype=np.float32),
                    "count":    0,
                })
                slot["mask_sum"] += union_orig
                slot["count"] += 1

            if verbose:
                kept = sum(
                    e["num_detections"]
                    for k, e in pass_result.items()
                    if isinstance(k, int)
                )
                print(f"   [{tta_name:8s}] raw detections this pass: {kept}")

        # Aggregate → binarise → connected components → instances
        final: dict = {}
        for q_idx, slot in accum.items():
            n = max(slot["count"], 1)
            if aggregate == "mean":
                agg = slot["mask_sum"] / n
            elif aggregate == "max":
                agg = (slot["mask_sum"] > 0).astype(np.float32)
            else:  # vote
                agg = (slot["mask_sum"] / n >= 0.5).astype(np.float32)

            binary = (agg >= threshold).astype(np.uint8)

            if do_pp and binary.sum() > 0:
                binary = self._postprocess_mask(binary > 0).astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            boxes, scores, instance_masks = [], [], []
            for lbl in range(1, num_labels):
                x, y, w, h, area = stats[lbl]
                if area < min_component_area:
                    continue
                m = (labels == lbl)
                boxes.append([float(x), float(y), float(x + w), float(y + h)])
                scores.append(float(agg[m].mean()))
                instance_masks.append(m)

            if instance_masks:
                entry = {
                    "prompt":         slot["prompt"],
                    "boxes":          np.array(boxes, dtype=np.float32),
                    "scores":         np.array(scores, dtype=np.float32),
                    "masks":          np.stack(instance_masks, axis=0),
                    "num_detections": len(instance_masks),
                    "tta_aggregate":  agg,
                }
            else:
                entry = {
                    "prompt":         slot["prompt"],
                    "boxes":          None,
                    "scores":         None,
                    "masks":          None,
                    "num_detections": 0,
                    "tta_aggregate":  agg,
                }

            if self.use_skeleton and entry["masks"] is not None:
                entry["skeleton"] = self._skeletonize_mask(entry["masks"])

            final[q_idx] = entry
            if verbose:
                print(f"   '{slot['prompt']}': {entry['num_detections']} components after TTA "
                      f"(union mask: {int(binary.sum())} px"
                      + (", postprocessed" if do_pp else "") + ")")

        final["_image"] = pil_raw
        final["_image_preprocessed"] = (
            self._preprocess(pil_raw) if apply_preprocess else pil_raw
        )
        return final

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        # Colors for different prompts
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        total_detections = 0

        # Draw results for each prompt
        for idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
            result = results[idx]
            prompt = result['prompt']
            color = colors[idx % len(colors)]

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            # In sliding-window mode `masks` is a single union mask of shape
            # (1, H, W) even when num_detections > 1. Guard against indexing
            # past the mask array — in that case draw the shared mask once.
            n_masks = len(masks) if masks is not None else 0
            shared_mask_mode = (
                masks is not None
                and n_masks == 1
                and result['num_detections'] > 1
            )
            mask_drawn = False  # prevents re-rendering the shared union mask per box

            for i in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None and not (shared_mask_mode and mask_drawn):
                    # Pick which mask to draw: per-instance if shapes match,
                    # else fall back to the shared union mask (index 0).
                    m_idx = i if i < n_masks else 0
                    mask = masks[m_idx]
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use different colors for different prompts
                    if color == 'red':
                        colored_mask[mask] = [1, 0, 0, 0.4]
                    elif color == 'blue':
                        colored_mask[mask] = [0, 0, 1, 0.4]
                    elif color == 'green':
                        colored_mask[mask] = [0, 1, 0, 0.4]
                    else:
                        colored_mask[mask] = [1, 1, 0, 0.4]
                    ax.imshow(colored_mask)
                    mask_drawn = True

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[i] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

        ax.axis('off')

        # Add title with all prompts
        prompts_str = ", ".join([f'"{results[k]["prompt"]}"' for k in sorted([k for k in results.keys() if not str(k).startswith('_')])])
        plt.suptitle(f'Text Prompts: {prompts_str}', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n✅ Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Detection confidence threshold (default: 0.30, lowered from 0.5 "
             "per thesis report §6 for better crack recall)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.30,
        help="NMS IoU threshold (default: 0.30, lowered from 0.5 so closely "
             "spaced crack segments can co-exist)"
    )

    # CLAHE preprocessing
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing (enabled by default per thesis §6)"
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=3.0,
        help="CLAHE clip limit (default: 3.0)"
    )
    parser.add_argument(
        "--clahe-grid",
        type=int,
        default=8,
        help="CLAHE tile grid size NxN (default: 8)"
    )

    # Debug / inspection
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save the CLAHE-preprocessed image next to the output "
             "(e.g. crack.png → crack_preprocessed.png) for debugging"
    )

    # Sliding-window inference
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Run sliding-window inference on tiles of the full-resolution image"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Sliding-window tile size in pixels (default: 640)"
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Sliding-window fractional overlap (default: 0.25 → stride=480 "
             "at tile_size=640)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar during sliding-window inference"
    )

    # Test-time augmentation (Tier 1, item 2 in next-steps roadmap)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Run test-time augmentation: average mask predictions across "
             "geometric flip / rotation variants. Free recall boost without "
             "retraining. Mutually exclusive with --sliding-window for now."
    )
    parser.add_argument(
        "--tta-transforms",
        nargs="+",
        default=["identity", "hflip", "vflip", "rot180"],
        choices=["identity", "hflip", "vflip", "rot180"],
        help="Geometric transforms to ensemble (default: all 4)"
    )
    parser.add_argument(
        "--tta-aggregate",
        choices=["mean", "max", "vote"],
        default="mean",
        help="How to combine per-pass union masks: mean (probabilistic), "
             "max (any pass), vote (majority). Default: mean."
    )
    parser.add_argument(
        "--tta-threshold",
        type=float,
        default=0.5,
        help="Binarisation threshold on the aggregated TTA map (default: 0.5)"
    )
    parser.add_argument(
        "--tta-min-area",
        type=int,
        default=5,
        help="Drop connected components smaller than this many pixels after "
             "TTA aggregation (default: 5)"
    )

    # Morphological post-processing (thesis report §4.3)
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Enable morphological post-processing (MORPH_CLOSE with line "
             "kernels to bridge crack gaps, then MORPH_OPEN to remove noise). "
             "For sliding-window inference this is applied ONCE on the final "
             "union mask so gaps across tile boundaries can be bridged."
    )
    parser.add_argument(
        "--pp-close",
        type=int,
        default=20,
        help="Post-processing line-kernel length for MORPH_CLOSE (default: 20)"
    )
    parser.add_argument(
        "--pp-open",
        type=int,
        default=3,
        help="Post-processing square-kernel size for MORPH_OPEN (default: 3). "
             "Set 0 to disable the opening step."
    )
    parser.add_argument(
        "--skeletonize",
        action="store_true",
        help="Also compute a 1-pixel skeleton (centerline) of the final mask. "
             "Exposed in results[q]['skeleton']. Useful for downstream "
             "length / orientation measurement (thesis report §4.3)."
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch / numpy / random, and enable cudnn.deterministic "
             "(default: 0). Set to make runs bit-reproducible."
    )

    args = parser.parse_args()

    # ---- Reproducibility (thesis report §4.3 / §6) ---------------------
    # Do this BEFORE any tensor allocation, model build, or dataloader
    # shuffling so every downstream op sees the same seeded state.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Seed set to {args.seed} (cudnn.deterministic=True)")

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        use_clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid=args.clahe_grid,
        use_postprocess=args.postprocess,
        pp_close_kernel=args.pp_close,
        pp_open_kernel=args.pp_open,
        use_skeleton=args.skeletonize,
    )

    # Route to one of: TTA, sliding-window, or single-pass inference.
    if args.tta and args.sliding_window:
        raise SystemExit("--tta and --sliding-window can't be combined yet. "
                         "Pick one for now.")
    if args.tta:
        results = inferencer.predict_tta(
            args.image, args.prompt,
            tta_transforms=tuple(args.tta_transforms),
            aggregate=args.tta_aggregate,
            threshold=args.tta_threshold,
            min_component_area=args.tta_min_area,
        )
    elif args.sliding_window:
        results = inferencer.predict_sliding_window(
            args.image, args.prompt,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            show_progress=not args.no_progress,
        )
    else:
        results = inferencer.predict(args.image, args.prompt)

    # Visualize
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks
    )

    # Optional: save the CLAHE-preprocessed image for debug / comparison
    if args.save_preprocessed:
        if not inferencer.use_clahe:
            print("\n⚠️  --save-preprocessed requested but CLAHE is disabled "
                  "(--no-clahe). Saving a copy of the raw image instead.")

        # Both predict() and predict_sliding_window() cache the
        # preprocessed image in results["_image_preprocessed"], so this is
        # always exactly what the model saw (not a re-computation).
        if "_image_preprocessed" in results:
            pre_img = results["_image_preprocessed"]
        else:
            # Fallback — shouldn't happen with current code paths, but guard
            # in case predict() is called in a way that skips storing it.
            pre_img = inferencer._preprocess(results["_image"])

        base, ext = os.path.splitext(args.output)
        if not ext:
            ext = ".png"
        pre_path = f"{base}_preprocessed{ext}"
        pre_img.save(pre_path)
        print(f"💾 Saved preprocessed image: {pre_path}")

    # Print summary
    print("\n" + "="*60)
    print("📊 Summary:")
    for idx in sorted([k for k in results.keys() if not str(k).startswith('_')]):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result['num_detections'] > 0 and result['scores'] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
        if result.get('masks') is not None:
            mask_px = int(np.asarray(result['masks']).sum())
            print(f"      Mask pixels: {mask_px}")
        if result.get('skeleton') is not None:
            skel_px = int(np.asarray(result['skeleton']).sum())
            print(f"      Skeleton pixels (centerline length ≈): {skel_px}")
    print("="*60)


if __name__ == "__main__":
    main()
