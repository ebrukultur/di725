#!/usr/bin/env python
"""
auair_yolos_manual_metrics_final.py – Fine‑tune YOLOS‑Tiny on the AU‑AIR dataset with manual mAP post‑processing and W&B logging.

* Converts AU‑AIR JSON → COCO JSON
* Trains/evaluates with Albumentations aug, Torchmetrics mAP
* Logs training and evaluation metrics to Weights & Biases
* Saves train and validation datasets as JSON files during training
"""

import os
import json
import random
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import cv2
import albumentations as A
from pycocotools.coco import COCO

import wandb  # Weights & Biases
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


def convert_to_coco(src_json: str, img_root: str, dst_json: str):
    src = json.load(open(src_json))
    images, anns, ann_id = [], [], 1
    for img_id, rec in enumerate(src["annotations"]):
        fname = rec["image_name"]
        w, h = Image.open(os.path.join(img_root, fname)).size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for bb in rec["bbox"]:
            x, y, bw, bh = bb["left"], bb["top"], bb["width"], bb["height"]
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
    cats = [{"id": i+1, "name": n} for i,n in enumerate(CLASSES)]
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f, indent=2)
    print(f"COCO saved: {dst_json} | images={len(images)} boxes={len(anns)}")


def save_subset_coco(dataset: Subset, coco: COCO, dst_json: str):
    """Save a subset of the dataset as a COCO-format JSON file."""
    subset_ids = [dataset.dataset.ids[i] for i in dataset.indices]
    images = [coco.loadImgs(img_id)[0] for img_id in subset_ids]
    anns = []
    for img_id in subset_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns.extend(coco.loadAnns(ann_ids))
    cats = [{"id": i+1, "name": n} for i, n in enumerate(CLASSES)]
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f, indent=2)
    print(f"Saved {dst_json} | images={len(images)} boxes={len(anns)}")


def build_transforms(img_size: int, train: bool = True):
    extra = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.2),
        A.HueSaturationValue(10, 15, 10, p=0.2),
    ] if train else []
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
            *extra,
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.01,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


def _strip_batch(obj):
    if torch.is_tensor(obj): return obj.squeeze(0)
    if isinstance(obj, list):    return _strip_batch(obj[0])
    if isinstance(obj, dict):    return {k: _strip_batch(v) for k,v in obj.items()}
    return obj


class AuAirDataset(Dataset):
    def __init__(self, img_root: str, ann_json: str, processor, tform):
        self.img_root, self.proc, self.tform = img_root, processor, tform
        self.coco = COCO(ann_json)
        self.ids  = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta   = self.coco.loadImgs(img_id)[0]
        img    = np.array(Image.open(os.path.join(self.img_root, meta["file_name"])).convert("RGB"))
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            if w <= 1 or h <= 1: continue
            boxes.append([x, y, x+w, y+h])
            labels.append(a["category_id"] - 1)

        if not boxes:
            return self.__getitem__((idx+1) % len(self))

        if self.tform:
            aug = self.tform(image=img, bboxes=boxes, category_ids=labels)
            img, boxes, labels = aug["image"], aug["bboxes"], aug["category_ids"]

        ann_list = []
        for (x0,y0,x1,y1), lab in zip(boxes, labels):
            bw, bh = x1 - x0, y1 - y0
            ann_list.append({
                "bbox": [x0, y0, bw, bh],
                "category_id": int(lab),
                "area": float(bw * bh),
                "iscrowd": 0,
            })

        enc = self.proc(
            Image.fromarray(img),
            annotations=[{"image_id": img_id, "annotations": ann_list}],
            return_tensors="pt",
        )
        return {k: _strip_batch(v) for k,v in enc.items()}


class SimpleCollator:
    def __call__(self, batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels":        [b.get("labels") for b in batch],
        }


def build_metric_fn(img_size: int):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    def fn(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, dict):
            logits, pred_boxes = preds["logits"], preds["pred_boxes"]
        else:
            logits, pred_boxes = preds[-2], preds[-1]
        if isinstance(logits, np.ndarray): logits = torch.from_numpy(logits)
        if isinstance(pred_boxes, np.ndarray): pred_boxes = torch.from_numpy(pred_boxes)
        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)
        batch,nq,_ = logits.shape
        cx = pred_boxes[...,0]*img_size; cy = pred_boxes[...,1]*img_size
        w  = pred_boxes[...,2]*img_size; h  = pred_boxes[...,3]*img_size
        x0= cx-0.5*w; y0= cy-0.5*h; x1= cx+0.5*w; y1= cy+0.5*h
        boxes_abs = torch.stack([x0,y0,x1,y1], dim=-1)
        p_list=[]
        for i in range(batch):
            mask=scores[i]>0
            p_list.append({"boxes":boxes_abs[i][mask].cpu(),"scores":scores[i][mask].cpu(),"labels":labels[i][mask].long().cpu()})
        gts=eval_pred.label_ids; g_list=[]
        if isinstance(gts, dict):
            for i in range(batch):
                b=l=torch.tensor([ [x,y,x+w0,y+h0] for x,y,w0,h0 in gts["boxes"][i] ])
                l=torch.tensor(gts["class_labels"][i])
                g_list.append({"boxes":b.cpu(),"labels":l.cpu()})
        else:
            for gt in gts:
                b=torch.tensor([ [x,y,x+w0,y+h0] for x,y,w0,h0 in gt["boxes"] ])
                l=torch.tensor(gt["class_labels"])
                g_list.append({"boxes":b.cpu(),"labels":l.cpu()})
        metric.update(p_list, g_list)
        out=metric.compute(); metric.reset()
        return {"mAP":out["map"].item(), "mAP.50":out["map_50"].item()}
    return fn


def main():
    parser=argparse.ArgumentParser()
    subs=parser.add_subparsers(dest="cmd", required=True)
    c=subs.add_parser("convert")
    c.add_argument("--ann",required=True); c.add_argument("--img-root",required=True); c.add_argument("--out",required=True)
    t=subs.add_parser("train")
    t.add_argument("--img-root",required=True); t.add_argument("--ann",required=True)
    t.add_argument("--img-size",type=int,default=384); t.add_argument("--epochs",type=int,default=10)
    t.add_argument("--batch",type=int,default=2); t.add_argument("--workers",type=int,default=0)
    t.add_argument("--outdir",default="yolos-auair")
    e=subs.add_parser("evaluate")
    e.add_argument("--img-root",required=True); e.add_argument("--ann",required=True)
    e.add_argument("--img-size",type=int,default=384); e.add_argument("--batch",type=int,default=1)
    e.add_argument("--workers",type=int,default=0); e.add_argument("--checkpoint",required=True)
    e.add_argument("--outdir",default="yolos-auair")
    args=parser.parse_args()

    # Initialize W&B
    wandb.init(project="auair-yolos", config=vars(args), name=f"{args.cmd}-{random.randint(0,9999)}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    processor=AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    if args.cmd=="convert":
        convert_to_coco(args.ann,args.img_root,args.out); return

    if args.cmd=="train":
        ds=AuAirDataset(args.img_root,args.ann,processor,build_transforms(args.img_size,True))
        idxs=list(range(len(ds))); random.shuffle(idxs); split=int(0.85*len(idxs))
        train_ds,val_ds=Subset(ds,idxs[:split]),Subset(ds,idxs[split:])
        
        # Save train and val datasets as JSON
        save_subset_coco(train_ds, ds.coco, os.path.join(args.outdir, "train.json"))
        save_subset_coco(val_ds, ds.coco, os.path.join(args.outdir, "val.json"))
        
        model=AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny", return_dict=True, num_labels=len(CLASSES), ignore_mismatched_sizes=True
        )
        wandb.watch(model)
        tr_args=TrainingArguments(
            output_dir=args.outdir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            num_train_epochs=args.epochs,
            dataloader_num_workers=args.workers,
            report_to=["wandb"],
            logging_steps=10,
            save_strategy="epoch"
        )
        trainer=Trainer(
            model=model,args=tr_args,
            train_dataset=train_ds,eval_dataset=val_ds,
            data_collator=SimpleCollator(),
            compute_metrics=build_metric_fn(args.img_size)
        )
        trainer.train(); trainer.save_model(args.outdir)

    else:
        val_ds=AuAirDataset(args.img_root,args.ann,processor,build_transforms(args.img_size,False))
        model=AutoModelForObjectDetection.from_pretrained(
            args.checkpoint, return_dict=True, num_labels=len(CLASSES), ignore_mismatched_sizes=True
        )
        ev_args=TrainingArguments(
            output_dir=args.outdir,
            per_device_eval_batch_size=args.batch,
            dataloader_num_workers=args.workers,
            report_to=["wandb"]
        )
        trainer=Trainer(
            model=model,args=ev_args,
            eval_dataset=val_ds,
            data_collator=SimpleCollator(),
            compute_metrics=build_metric_fn(args.img_size)
        )
        metrics=trainer.evaluate()
        print("Evaluation metrics:",metrics)
        # Log metrics
        wandb.log(metrics)

    wandb.finish()

if __name__=="__main__": main()