#!/usr/bin/env python
"""
auair_yolos.py  ‑‑  fine‑tune YOLOS‑Tiny on the AU‑AIR dataset

* Converts AU‑AIR native JSON → COCO JSON
* Trains/evaluates with Albumentations aug and Torchmetrics mAP
* Fits in ~4 GB VRAM
"""

from __future__ import annotations
import os, json, random, argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import cv2, albumentations as A
from pycocotools.coco import COCO

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

CLASSES = [
    "person", "car", "truck", "van",
    "motorbike", "bicycle", "bus", "trailer"
]

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  CONVERTER  (AU‑AIR native ➜ COCO detection JSON)
# ─────────────────────────────────────────────────────────────────────────────
def convert_to_coco(src_json: str, img_root: str, dst_json: str):
    src = json.load(open(src_json))
    images, anns, ann_id = [], [], 1
    for img_id, rec in enumerate(src["annotations"]):
        fname = rec["image_name"]
        w, h  = Image.open(os.path.join(img_root, fname)).size
        images.append({"id": img_id,
                       "file_name": fname,
                       "width": w,
                       "height": h})
        for bb in rec["bbox"]:
            x, y = bb["left"], bb["top"]
            bw, bh = bb["width"], bb["height"]
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bb["class"] + 1,      # COCO → 1‑based
                "bbox": [x, y, bw, bh],              # xywh
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    json.dump({"images": images,
               "annotations": anns,
               "categories": cats},
              open(dst_json, "w"))
    print(f"✅  COCO saved: {dst_json}  | imgs={len(images)}  boxes={len(anns)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  AUGMENTATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms(img_size: int, train: bool = True):
    extra = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.2),
        A.HueSaturationValue(10, 15, 10, p=0.2),
    ] if train else []
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                img_size, img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)                # YOLO‑style gray padding
            ),
            *extra,
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"]
        )
    )

# ─────────────────────────────────────────────────────────────────────────────
# helper – strip the batch dim that AutoImageProcessor adds
# ─────────────────────────────────────────────────────────────────────────────
def _strip_batch(obj):
    """Tensor → squeeze(0); single‑element list → elem; dict → recurse."""
    if torch.is_tensor(obj):
        return obj.squeeze(0)
    if isinstance(obj, list):
        return _strip_batch(obj[0])
    if isinstance(obj, dict):
        return {k: _strip_batch(v) for k, v in obj.items()}
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class AuAirDataset(Dataset):
    def __init__(self, img_root: str, ann_json: str, processor, tform):
        self.img_root, self.proc, self.tform = img_root, processor, tform
        self.coco = COCO(ann_json)
        self.ids  = list(self.coco.imgs.keys())

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta   = self.coco.loadImgs(img_id)[0]

        img = np.array(
            Image.open(os.path.join(self.img_root, meta["file_name"]))
                 .convert("RGB")
        )

        # ---- COCO boxes -----------------------------------------------------
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes_xyxy, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])        # xyxy
            labels.append(a["category_id"] - 1)            # 0‑based

        # ---- Albumentations aug --------------------------------------------
        if self.tform:
            aug = self.tform(image=img,
                             bboxes=boxes_xyxy,
                             category_ids=labels)
            img        = aug["image"]
            boxes_xyxy = aug["bboxes"]
            labels     = aug["category_ids"]

        # ---- COCO‑style target for the processor ---------------------------
        annotations = []
        for box, lab in zip(boxes_xyxy, labels):
            x1, y1, x2, y2 = box
            annotations.append({
                "bbox":       [x1, y1, x2 - x1, y2 - y1],   # xywh
                "category_id": int(lab),
                "area":        float((x2 - x1) * (y2 - y1)),
                "iscrowd":     0,
            })
        target = {"image_id": img_id, "annotations": annotations}

        enc = self.proc(
            Image.fromarray(img),
            annotations=[target],
            return_tensors="pt"
        )
        return {k: _strip_batch(v) for k, v in enc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  COLLATOR  (simple and robust)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCollator:
    """Stack already‑padded images; keep targets as list of dicts."""
    def __call__(self, batch):
        pixel_values = torch.stack([s["pixel_values"] for s in batch])  # (B,3,H,W)
        labels       = [s["labels"] for s in batch]                     # list of dicts
        return {"pixel_values": pixel_values, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  METRICS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_metric_fn(processor):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def fn(eval_preds):
        preds, gts = eval_preds
        sizes = [t["orig_size"] for t in gts]        # (H, W) per sample

        processed = processor.post_process_object_detection(
            preds, threshold=0.0, target_sizes=sizes
        )

        p_list, g_list = [], []
        for p, gt in zip(processed, gts):
            p_list.append({
                "boxes":  p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu(),
            })
            g_list.append({
                "boxes":  gt["boxes"].cpu(),
                "labels": gt["class_labels"].cpu(),
            })
        metric.update(p_list, g_list)
        out = metric.compute()
        metric.reset()
        return {
            "mAP":    out["map"].item(),
            "mAP.50": out["map_50"].item(),
        }
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# 6️⃣  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    c = sp.add_parser("convert")
    c.add_argument("--ann")
    c.add_argument("--img-root")
    c.add_argument("--out")

    t = sp.add_parser("train")
    t.add_argument("--img-root")
    t.add_argument("--ann")
    t.add_argument("--img-size", type=int, default=512)
    t.add_argument("--epochs",   type=int, default=20)
    t.add_argument("--batch",    type=int, default=1)
    t.add_argument("--outdir",   default="yolos-auair")

    args = p.parse_args()

    # ── convert only ────────────────────────────────────────────────────────
    if args.cmd == "convert":
        convert_to_coco(args.ann, args.img_root, args.out)
        return

    # ── prepare training ────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    proc  = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained(
        "hustvl/yolos-tiny",
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    )

    full_ds = AuAirDataset(
        args.img_root, args.ann, proc,
        build_transforms(args.img_size, train=True)
    )
    idxs = list(range(len(full_ds))); random.shuffle(idxs)
    split = int(0.85 * len(idxs))
    train_ds = Subset(full_ds, idxs[:split])

    val_ds = Subset(
        AuAirDataset(
            args.img_root, args.ann, proc,
            build_transforms(args.img_size, train=False)
        ),
        idxs[split:]
    )

    collator = SimpleCollator()

    tr_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 4 // args.batch),   # logical BS≈4
        learning_rate=2e-5,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to=[],              # set to ["wandb"] if you use WandB
    )

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=build_metric_fn(proc),
    )
    trainer.train()
    trainer.save_model(args.outdir)


if __name__ == "__main__":
    main()
