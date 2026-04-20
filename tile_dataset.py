"""Tile-based COCO dataset for SAM3 LoRA crack segmentation.

Hairline cracks (1-3 px wide in the camera image) become sub-pixel after a
whole-wall photo is downscaled to the model resolution. Tiling preserves
per-pixel crack signal: a 1008x1008 tile cropped from the native-resolution
photo keeps every crack pixel intact.

Drop-in replacement for COCOSegmentDataset — returns the same Datapoint shape.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2

import pycocotools.mask as mask_utils
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


class WeightedDistributedSampler(Sampler):
    """Weighted sampler that also works under DDP.

    Each epoch the rank-0 generator draws `total_size` indices with replacement
    from a shared weight distribution, then the list is sliced per rank so every
    GPU sees a disjoint shard. When `num_replicas=1` it degenerates to a plain
    `WeightedRandomSampler` — so the same code path covers single-GPU runs.
    """

    def __init__(
        self,
        weights,
        num_samples: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range [0, {num_replicas})")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if (self.weights < 0).any():
            raise ValueError("sampler weights must be non-negative")
        if float(self.weights.sum()) <= 0:
            raise ValueError("sampler weights sum to zero")

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.total_size = (num_samples // num_replicas) * num_replicas
        if self.total_size == 0:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= num_replicas ({num_replicas})"
            )
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(self.seed) + int(self.epoch))
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        ).tolist()
        shard = indices[self.rank:self.total_size:self.num_replicas]
        assert len(shard) == self.num_samples
        return iter(shard)

    def __len__(self) -> int:
        return self.num_samples


class TiledCOCODataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tile_size: int = 1008,
        overlap: float = 0.25,
        min_crack_pixels: int = 0,
        random_offset: bool = True,
        augment: bool = True,
        image_cache_size: int = 8,
    ):
        """
        Args:
            data_dir:          Root directory containing train/valid/test/.
            split:             'train', 'valid', or 'test'.
            tile_size:         Tile edge length (must equal SAM3 input resolution).
            overlap:           Fractional overlap between adjacent tiles (0.0-0.5).
            min_crack_pixels:  Drop tiles whose union mask has fewer than this many
                               foreground pixels. 0 = keep all tiles (incl. negatives).
            random_offset:     Jitter tile origin per epoch (training only).
            augment:           Apply flip / rot90 / color jitter (training only).
            image_cache_size:  Number of decoded source images to keep in RAM.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_crack_pixels = min_crack_pixels
        self.random_offset = random_offset and (split == "train")
        self.augment = augment and (split == "train")

        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {c["id"]: c["name"] for c in self.coco_data["categories"]}

        self.img_to_anns: dict = {}
        for ann in self.coco_data["annotations"]:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.tile_specs: List[Tuple[int, int, int]] = self._build_tile_index()

        self._image_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._image_cache_size = image_cache_size

        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self._normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self._color_jitter = v2.ColorJitter(brightness=0.25, hue=0.07)

        print(f"Loaded TiledCOCODataset: {split} split")
        print(f"  Source images:       {len(self.images)}")
        print(f"  Annotations:         {len(self.coco_data['annotations'])}")
        print(f"  Tiles ({tile_size}px, {int(overlap*100)}% overlap): {len(self.tile_specs)}")
        if min_crack_pixels > 0:
            print(f"  (filtered by min_crack_pixels={min_crack_pixels})")
        print(f"  Categories:          {self.categories}")

    def _tile_origins(self, w: int, h: int) -> List[Tuple[int, int]]:
        stride = max(1, int(round(self.tile_size * (1.0 - self.overlap))))
        if w <= self.tile_size:
            xs = [0]
        else:
            xs = list(range(0, w - self.tile_size + 1, stride))
            if xs[-1] != w - self.tile_size:
                xs.append(w - self.tile_size)
        if h <= self.tile_size:
            ys = [0]
        else:
            ys = list(range(0, h - self.tile_size + 1, stride))
            if ys[-1] != h - self.tile_size:
                ys.append(h - self.tile_size)
        return [(x, y) for y in ys for x in xs]

    def _decode_ann_mask(self, ann: dict, h: int, w: int) -> np.ndarray:
        seg = ann.get("segmentation")
        if seg is None:
            return np.zeros((h, w), dtype=np.uint8)
        if isinstance(seg, dict):
            return mask_utils.decode(seg).astype(np.uint8)
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)
        return np.zeros((h, w), dtype=np.uint8)

    def _build_tile_index(self) -> List[Tuple[int, int, int]]:
        specs: List[Tuple[int, int, int]] = []
        self.tile_crack_pixels: List[int] = []
        union_cache: dict = {}

        for img_id in sorted(self.images.keys()):
            info = self.images[img_id]
            w, h = info["width"], info["height"]

            origins = self._tile_origins(w, h)

            if img_id not in union_cache:
                union = np.zeros((h, w), dtype=np.uint8)
                for ann in self.img_to_anns.get(img_id, []):
                    union |= self._decode_ann_mask(ann, h, w)
                union_cache[img_id] = union
            union = union_cache[img_id]

            for x, y in origins:
                tile = union[y:y + self.tile_size, x:x + self.tile_size]
                pixels = int(tile.sum())
                if pixels >= self.min_crack_pixels:
                    specs.append((img_id, x, y))
                    self.tile_crack_pixels.append(pixels)

        return specs

    def compute_tile_weights(
        self,
        num_bins: int = 5,
        power: float = 1.0,
        empty_tile_weight: float = 0.1,
        verbose: bool = True,
    ) -> np.ndarray:
        """Log-space binned inverse-frequency weights over tiles.

        Strategy: positive tiles (with any crack pixel) are binned by
        log(1+pixels) into `num_bins` equal-width bins; each tile gets
        weight = (1 / bin_count)^power so rare (dense) bins are upweighted.
        Empty tiles (0 crack pixels) receive `empty_tile_weight * mean_positive`
        so negatives are not oversampled.

        Returns a numpy array of shape (len(dataset),), non-negative.
        """
        pixels = np.asarray(self.tile_crack_pixels, dtype=np.float64)
        if pixels.size != len(self.tile_specs):
            raise RuntimeError(
                "tile_crack_pixels is out of sync with tile_specs; "
                "rebuild dataset"
            )

        weights = np.zeros_like(pixels)
        pos_mask = pixels > 0
        empty_mask = ~pos_mask

        bin_counts = np.zeros(num_bins, dtype=np.int64)
        if pos_mask.any():
            log_pix = np.log1p(pixels[pos_mask])
            lo = float(log_pix.min())
            hi = float(log_pix.max()) + 1e-9
            if hi <= lo:
                hi = lo + 1e-6
            edges = np.linspace(lo, hi, num_bins + 1)
            bin_ids = np.digitize(log_pix, edges[1:-1])
            bin_ids = np.clip(bin_ids, 0, num_bins - 1)

            bin_counts = np.bincount(bin_ids, minlength=num_bins)
            bin_weights = np.zeros(num_bins, dtype=np.float64)
            nonzero = bin_counts > 0
            bin_weights[nonzero] = (1.0 / bin_counts[nonzero]) ** power

            weights_pos = bin_weights[bin_ids]
            weights[pos_mask] = weights_pos
            mean_pos = float(weights_pos.mean()) if weights_pos.size else 1.0
        else:
            mean_pos = 1.0

        if empty_mask.any() and empty_tile_weight > 0:
            weights[empty_mask] = mean_pos * float(empty_tile_weight)

        if float(weights.sum()) <= 0:
            weights[:] = 1.0

        if verbose:
            print(f"  Sampler weights (num_bins={num_bins}, power={power}, "
                  f"empty_tile_weight={empty_tile_weight}):")
            print(f"    positive tiles: {int(pos_mask.sum())}  "
                  f"empty tiles: {int(empty_mask.sum())}")
            if pos_mask.any():
                print(f"    positive bin counts (low→high density): "
                      f"{bin_counts.tolist()}")
                # expected share of draws per bin (positive only)
                bin_w = np.zeros(num_bins)
                nz = bin_counts > 0
                bin_w[nz] = (1.0 / bin_counts[nz]) ** power
                total_pos_w = float((bin_w * bin_counts).sum())
                if total_pos_w > 0:
                    share = (bin_w * bin_counts) / total_pos_w
                    print(f"    expected positive draw share per bin: "
                          f"{np.round(share, 3).tolist()}")

        return weights

    def _load_image_bgr(self, img_id: int) -> np.ndarray:
        if img_id in self._image_cache:
            self._image_cache.move_to_end(img_id)
            return self._image_cache[img_id]

        info = self.images[img_id]
        path = self.split_dir / info["file_name"]
        pil = PILImage.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        self._image_cache[img_id] = bgr
        if len(self._image_cache) > self._image_cache_size:
            self._image_cache.popitem(last=False)
        return bgr

    def _crop_or_pad(self, arr: np.ndarray, x0: int, y0: int, fill: int) -> np.ndarray:
        h, w = arr.shape[:2]
        ts = self.tile_size

        x_end = min(x0 + ts, w)
        y_end = min(y0 + ts, h)
        crop = arr[y0:y_end, x0:x_end]

        ch = y_end - y0
        cw = x_end - x0
        if ch == ts and cw == ts:
            return crop

        if arr.ndim == 3:
            out = np.full((ts, ts, arr.shape[2]), fill, dtype=arr.dtype)
        else:
            out = np.full((ts, ts), fill, dtype=arr.dtype)
        out[:ch, :cw] = crop
        return out

    def _enhance_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        out = out.astype(np.float32)
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.3, 0, 255)
        out[:, :, 0] = np.clip(out[:, :, 0] * 1.2, 0, 255)
        out = out.astype(np.uint8)

        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 2.5, blur, -1.5, 0)
        return out

    def __len__(self) -> int:
        return len(self.tile_specs)

    def __getitem__(self, idx: int) -> Datapoint:
        img_id, x0, y0 = self.tile_specs[idx]
        info = self.images[img_id]
        orig_h, orig_w = info["height"], info["width"]

        if self.random_offset:
            stride = max(1, int(round(self.tile_size * (1.0 - self.overlap))))
            jitter = stride // 4
            if jitter > 0:
                x0 = int(np.clip(x0 + np.random.randint(-jitter, jitter + 1),
                                 0, max(0, orig_w - self.tile_size)))
                y0 = int(np.clip(y0 + np.random.randint(-jitter, jitter + 1),
                                 0, max(0, orig_h - self.tile_size)))

        img_bgr_full = self._load_image_bgr(img_id)
        mean_fill = int(img_bgr_full.mean())
        tile_bgr = self._crop_or_pad(img_bgr_full, x0, y0, fill=mean_fill)

        flip_h = self.augment and (np.random.rand() < 0.5)
        flip_v = self.augment and (np.random.rand() < 0.5)
        rot_k = int(np.random.randint(0, 4)) if self.augment else 0

        if flip_h:
            tile_bgr = np.ascontiguousarray(tile_bgr[:, ::-1])
        if flip_v:
            tile_bgr = np.ascontiguousarray(tile_bgr[::-1, :])
        if rot_k:
            tile_bgr = np.ascontiguousarray(np.rot90(tile_bgr, k=rot_k))

        tile_bgr = self._enhance_bgr(tile_bgr)
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        pil_tile = PILImage.fromarray(tile_rgb)

        if self.augment:
            pil_tile = self._color_jitter(pil_tile)

        image_tensor = self._normalize(pil_tile)

        objects: List[Object] = []
        object_class_names: List[str] = []
        ann_list = self.img_to_anns.get(img_id, [])

        for i, ann in enumerate(ann_list):
            full_mask = self._decode_ann_mask(ann, orig_h, orig_w)
            tile_mask = self._crop_or_pad(full_mask, x0, y0, fill=0)

            if flip_h:
                tile_mask = np.ascontiguousarray(tile_mask[:, ::-1])
            if flip_v:
                tile_mask = np.ascontiguousarray(tile_mask[::-1, :])
            if rot_k:
                tile_mask = np.ascontiguousarray(np.rot90(tile_mask, k=rot_k))

            ys, xs = np.where(tile_mask > 0)
            if ys.size == 0:
                continue

            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            w = float(x_max - x_min + 1)
            h = float(y_max - y_min + 1)
            cx = float(x_min) + w / 2.0
            cy = float(y_min) + h / 2.0

            box_tensor = torch.tensor([
                cx / self.tile_size,
                cy / self.tile_size,
                w  / self.tile_size,
                h  / self.tile_size,
            ], dtype=torch.float32)

            class_name = self.categories.get(ann.get("category_id", 0), "object")
            object_class_names.append(class_name)

            segment = torch.from_numpy(tile_mask > 0)

            objects.append(Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=len(objects),
                segment=segment,
            ))

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.tile_size, self.tile_size),
        )

        class_to_object_ids: dict = defaultdict(list)
        for obj, name in zip(objects, object_class_names):
            class_to_object_ids[name.lower()].append(obj.object_id)

        queries: List[FindQueryLoaded] = []
        if class_to_object_ids:
            for query_text, obj_ids in class_to_object_ids.items():
                queries.append(FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=0,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1,
                    ),
                ))
        else:
            queries.append(FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=0,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1,
                ),
            ))

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_tile],
        )
