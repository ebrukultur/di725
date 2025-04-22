#!/usr/bin/env python
"""
auair_yolos_map_evaluation_with_perclass.py  – Compute mAP on the AU‑AIR dataset using a fine-tuned YOLOS-Tiny with per-class AP breakdown.

"""
# Import required libraries
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import albumentations as A
import cv2
from pycocotools.coco import COCO
from types import SimpleNamespace

# Define class names for AU-AIR dataset
CLASSES = ["person","car","truck","van","motorbike","bicycle","bus","trailer"]

# Create image preprocessing pipeline
def build_transforms(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114))
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

# Dataset class for AU-AIR data
class AuAirDataset(Dataset):
    def __init__(self, img_root, ann_json, processor, img_size):
        self.img_root = img_root
        self.coco = COCO(ann_json)
        self.ids = list(self.coco.imgs.keys())
        self.proc = processor
        self.transforms = build_transforms(img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        bboxes, labels = [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            if w<=1 or h<=1: continue
            bboxes.append([x,y,x+w,y+h])
            labels.append(a["category_id"] - 1)
        transformed = self.transforms(image=img, bboxes=bboxes, category_ids=labels)
        img_t = transformed["image"]
        enc = self.proc(Image.fromarray(img_t), return_tensors="pt")
        # prepare gt
        gt_boxes = []
        gt_labels = []
        for (x0,y0,x1,y1), lab in zip(transformed["bboxes"], transformed["category_ids"]):
            gt_boxes.append([x0, y0, x1, y1])
            gt_labels.append(lab)
        gt = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float),
            "labels": torch.tensor(gt_labels, dtype=torch.long)
        }
        return enc["pixel_values"].squeeze(0), gt

# Main function to evaluate model
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root",   required=True)
    parser.add_argument("--ann",        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-size",   type=int, default=384)
    parser.add_argument("--batch",      type=int, default=1)
    parser.add_argument("--workers",    type=int, default=2)
    args = parser.parse_args()

    # Load YOLOS processor and model
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained(
        args.checkpoint,
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Create dataset and dataloader
    ds = AuAirDataset(args.img_root, args.ann, processor, args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=lambda x: x)

    # Initialize mAP metric
    metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    class_metrics=True          # ← this is the flag that turns on map_per_class
)


    # Evaluate model on dataset
    for batch in dl:
        imgs, gts = zip(*batch)
        pixel_values = torch.stack(imgs)
        if torch.cuda.is_available(): pixel_values = pixel_values.cuda()
        with torch.no_grad():
            outputs = model(pixel_values)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        target_sizes = [(args.img_size, args.img_size)] * pixel_values.shape[0]
        processed = processor.post_process_object_detection(
            SimpleNamespace(logits=logits, pred_boxes=pred_boxes),
            threshold=0.0, target_sizes=target_sizes
        )
        p_list, g_list = [], []
        for det, gt in zip(processed, gts):
            p_list.append({
                'boxes': det['boxes'].cpu(),
                'scores': det['scores'].cpu(),
                'labels': det['labels'].cpu()
            })
            g_list.append(gt)
        metric.update(p_list, g_list)
    # Compute and print mAP results
    res = metric.compute()
    # print overall
    print(f"Overall mAP (0.5:0.95): {res['map']:.4f}")
    print(f"Overall mAP@0.5:      {res['map_50']:.4f}\n")
    # per-class
    ap_per_class = res['map_per_class'].cpu().numpy()
    # Check array is at least 1D to avoid iteration errors
    ap_per_class = np.atleast_1d(ap_per_class)
    print("Per-class AP:")
    for cls_name, ap in zip(CLASSES, ap_per_class):
        print(f"  {cls_name:12s}: {ap:.4f}")
# Run main function
if __name__ == '__main__':
    main()
