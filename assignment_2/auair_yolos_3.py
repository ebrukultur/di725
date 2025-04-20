#!/usr/bin/env python
"""
auair_yolos.py  –  fine‑tune YOLOS‑Tiny (or ViT-based) on the AU‑AIR dataset

* Converts AU‑AIR native JSON → COCO JSON
* Splits dataset into 30,000 train‑val and 2,823 test samples
* Trains/evaluates with Albumentations aug, Torchmetrics per-class AP@0.5/mAP, and W&B logging

Usage examples:
  # 1. Convert annotations to COCO format
  python auair_yolos.py convert \
    --ann path/to/auair_native.json \
    --img-root path/to/images \
    --out path/to/auair_coco.json

  # 2. Train model (YOLOS‑Tiny)
  python auair_yolos.py train \
    --img-root path/to/images \
    --ann path/to/auair_coco.json \
    --img-size 384 \
    --epochs 20 \
    --batch 4 \
    --workers 4 \
    --outdir yolos-auair

  # 3. Evaluate model on test set
  python auair_yolos.py evaluate \
    --img-root path/to/images \
    --ann path/to/auair_coco.json \
    --model-dir yolos-auair \
    --img-size 384 \
    --batch 4 \
    --workers 4

Speed-up tips:
- Reduce `--img-size` (e.g., 256 or 320) to lower memory and compute.
- Lower `--epochs` to shorten training time.
- Increase `--batch` size if your GPU has headroom (fewer steps per epoch).
- Use `--no-eval` in `train` to skip validation loops.
- Adjust `--workers` for faster data loading.
"""

from __future__ import annotations
import os
import sys
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import cv2
import albumentations as A
from pycocotools.coco import COCO

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from transformers.integrations import WandbCallback
import wandb
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# eight traffic classes in AU‑AIR
CLASSES = [
    "person", "car", "truck", "van",
    "motorbike", "bicycle", "bus", "trailer"
]

# 1️⃣ Convert AU‑AIR native JSON → COCO-format JSON
def convert_to_coco(src_json: str, img_root: str, dst_json: str):
    src = json.load(open(src_json))
    images, anns, ann_id = [], [], 1
    for img_id, rec in enumerate(src["annotations"]):
        fname = rec["image_name"]
        w, h = Image.open(os.path.join(img_root, fname)).size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for bb in rec["bbox"]:
            x, y = bb["left"], bb["top"]
            bw, bh = bb["width"], bb["height"]
            if bw <= 1 or bh <= 1:
                continue
            anns.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": bb["class"] + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1
    cats = [{"id": i+1, "name": n} for i, n in enumerate(CLASSES)]
    with open(dst_json, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    print(f"✅ COCO saved: {dst_json} | images={len(images)} boxes={len(anns)}")

# 2️⃣ Data augmentations
def build_transforms(img_size: int, train: bool = True):
    extra = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.2),
        A.HueSaturationValue(10, 15, 10, p=0.2),
    ] if train else []
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
        *extra
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_visibility=0.01,
        clip=True,
        filter_invalid_bboxes=True
    ))

# strip batch dim from HF processor outputs
def _strip_batch(obj):
    if torch.is_tensor(obj):
        return obj.squeeze(0)
    if isinstance(obj, list):
        return _strip_batch(obj[0])
    if isinstance(obj, dict):
        return {k: _strip_batch(v) for k, v in obj.items()}
    return obj

# 3️⃣ Dataset wrapper
class AuAirDataset(Dataset):
    def __init__(self, img_root: str, ann_json: str, processor, tform):
        self.img_root = img_root
        self.proc = processor
        self.tform = tform
        self.coco = COCO(ann_json)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta = self.coco.loadImgs(img_id)[0]
        img = np.array(
            Image.open(os.path.join(self.img_root, meta["file_name"]))
                 .convert("RGB")
        )
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x+w, y+h])
            labels.append(a["category_id"]-1)
        if not boxes:
            return self.__getitem__((idx+1) % len(self.ids))
        if self.tform:
            aug = self.tform(image=img, bboxes=boxes, category_ids=labels)
            img = aug["image"]
            boxes = aug["bboxes"]
            labels = aug["category_ids"]
        targets = []
        for box, lab in zip(boxes, labels):
            x1, y1, x2, y2 = box
            targets.append({
                "bbox": [x1, y1, x2-x1, y2-y1],
                "category_id": int(lab),
                "area": float((x2-x1)*(y2-y1)),
                "iscrowd": 0,
            })
        enc = self.proc(
            Image.fromarray(img),
            annotations=[{"image_id": img_id, "annotations": targets}],
            return_tensors="pt"
        )
        return {k: _strip_batch(v) for k, v in enc.items()}

# 4️⃣ Collator
class SimpleCollator:
    def __call__(self, batch):
        pixel_values = torch.stack([s["pixel_values"] for s in batch])
        labels = [s["labels"] for s in batch]
        return {"pixel_values": pixel_values, "labels": labels}

# 5️⃣ Metric: per-class AP@IoU=0.5 and overall mAP
def build_metric_fn(processor):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.5])
    def fn(eval_preds):
        logits, pred_boxes = eval_preds
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(pred_boxes, np.ndarray):
            pred_boxes = torch.from_numpy(pred_boxes)
        sizes = [gt["orig_size"] for gt in eval_preds[1]]
        processed = processor.post_process_object_detection(
            {"logits": logits, "pred_boxes": pred_boxes},
            threshold=0.0,
            target_sizes=sizes
        )
        p_list, g_list = [], []
        for p, gt in zip(processed, eval_preds[1]):
            p_list.append({"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()})
            g_list.append({"boxes": gt["boxes"].cpu(), "labels": gt["class_labels"].cpu()})
        metric.update(p_list, g_list)
        out = metric.compute()
        metric.reset()
        results = {}
        per_class = out.get("map_per_class")
        if per_class is not None:
            for idx, cls in enumerate(CLASSES):
                results[f"AP/{cls}"] = per_class[idx].item()
        results["mAP"] = out["map"].item()
        return results
    return fn

# 6️⃣ MAIN: subcommands convert/train/evaluate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="cmd", required=True)
    # convert
    c = sp.add_parser("convert")
    c.add_argument("--ann", required=True)
    c.add_argument("--img-root", required=True)
    c.add_argument("--out", required=True)
    # train
    t = sp.add_parser("train")
    t.add_argument("--img-root", required=True)
    t.add_argument("--ann", required=True)
    t.add_argument("--img-size", type=int, default=384)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--outdir", default="yolos-auair")
    # evaluate
    e = sp.add_parser("evaluate")
    e.add_argument("--img-root", required=True)
    e.add_argument("--ann", required=True)
    e.add_argument("--model-dir", required=True)
    e.add_argument("--img-size", type=int, default=384)
    e.add_argument("--batch", type=int, default=4)
    e.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # reproducibility
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # convert only
    if args.cmd == "convert":
        convert_to_coco(args.ann, args.img_root, args.out)
        sys.exit(0)

    # load processor
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    if args.cmd == "train":
        wandb.init(project="auair-yolos", config=vars(args), name=f"train-{datetime.now():%Y%m%d-%H%M}")
        os.makedirs(args.outdir, exist_ok=True)
        model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny", num_labels=len(CLASSES), ignore_mismatched_sizes=True
        )
        full_ds = AuAirDataset(args.img_root, args.ann, processor, build_transforms(args.img_size, True))
        idxs = list(range(len(full_ds))); random.shuffle(idxs)
        test_size = 2823
        test_idxs = idxs[:test_size]
        rem_idxs = idxs[test_size:]
        val_size = int(0.1 * len(rem_idxs))
        val_idxs = rem_idxs[:val_size]
        train_idxs = rem_idxs[val_size:]
        train_ds = Subset(full_ds, train_idxs)
        val_ds   = Subset(AuAirDataset(args.img_root, args.ann, processor, build_transforms(args.img_size, False)), val_idxs)
        tr_args = TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            weight_decay=1e-4,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            report_to=["wandb"],
            dataloader_num_workers=args.workers
        )
        trainer = Trainer(
            model=model,
            args=tr_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=SimpleCollator(),
            compute_metrics=build_metric_fn(processor),
            callbacks=[WandbCallback]
        )
        trainer.train()
        trainer.save_model(args.outdir)

    elif args.cmd == "evaluate":
        model = AutoModelForObjectDetection.from_pretrained(
            args.model_dir, ignore_mismatched_sizes=True
        )
        full_ds = AuAirDataset(args.img_root, args.ann, processor, build_transforms(args.img_size, False))
        idxs = list(range(len(full_ds))); random.shuffle(idxs)
        test_idxs = idxs[:2823]
        test_ds = Subset(full_ds, test_idxs)
        eval_args = TrainingArguments(
            output_dir=args.model_dir,
            per_device_eval_batch_size=args.batch,
            dataloader_num_workers=args.workers
        )
        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=SimpleCollator(),
            compute_metrics=build_metric_fn(processor)
        )
        metrics = trainer.evaluate(eval_dataset=test_ds)
        print(metrics)
