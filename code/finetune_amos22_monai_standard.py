import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import finetune_spleen_monai_standard as base
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)


NUM_CLASSES = 16


def split_cases(data_root: str, val_count: int, seed: int) -> Tuple[List[dict], List[dict]]:
    root = Path(data_root)
    if (root / "amos22").exists():
        root = root / "amos22"
    train = collect_pairs(root / "imagesTr", root / "labelsTr")
    val = collect_pairs(root / "imagesVa", root / "labelsVa")
    if getattr(split_cases, "max_train", 0):
        train = train[: split_cases.max_train]
    if getattr(split_cases, "max_val", 0):
        val = val[: split_cases.max_val]
    if not train or not val:
        raise RuntimeError(f"No AMOS22 train/val pairs found under {root}")
    return train, val


split_cases.max_train = 0
split_cases.max_val = 0


def collect_pairs(image_dir: Path, label_dir: Path) -> List[dict]:
    labels = {p.name: p for p in label_dir.glob("*.nii.gz") if not p.name.startswith("._")}
    items = []
    for image in sorted(p for p in image_dir.glob("*.nii.gz") if not p.name.startswith("._")):
        label = labels.get(image.name)
        if label is not None:
            items.append({"image": str(image), "label": str(label), "case": image.name.replace(".nii.gz", "")})
    return items


def build_transforms(args: argparse.Namespace) -> Tuple[Compose, Compose]:
    pixdim = tuple(args.pixdim)
    roi_size = tuple(args.roi_size)
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if not getattr(args, "no_spacing", False):
        transforms.append(Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")))
    transforms += [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ]
    train = Compose(
        transforms
        + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=args.pos_ratio,
                neg=args.neg_ratio,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.1),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.1),
            RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val = Compose(transforms + [EnsureTyped(keys=["image", "label"])])
    return train, val


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=tuple(args.channels),
        strides=tuple(args.strides),
        num_res_units=args.num_res_units,
        norm="INSTANCE",
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device, args: argparse.Namespace, output_csv: Path | None = None) -> float:
    model.eval()
    post_steps = [
        AsDiscreted(keys="pred", argmax=True, to_onehot=NUM_CLASSES),
        AsDiscreted(keys="label", to_onehot=NUM_CLASSES),
    ]
    if args.keep_largest:
        post_steps.append(
            KeepLargestConnectedComponentd(
                keys="pred",
                applied_labels=list(range(1, NUM_CLASSES)),
                is_onehot=True,
                independent=True,
            )
        )
    post = Compose(post_steps)
    metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    rows = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = sliding_window_inference(
            image,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
            mode="gaussian",
        )
        data = [{"pred": p, "label": l} for p, l in zip(decollate_batch(logits), decollate_batch(label))]
        data = [post(d) for d in data]
        preds = torch.stack([d["pred"] for d in data])
        labels = torch.stack([d["label"] for d in data])
        metric(y_pred=preds, y=labels)
        case_dice = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True)(y_pred=preds, y=labels)
        row = {"case": batch.get("case", ["unknown"])[0], "dice": float(torch.nanmean(case_dice).item())}
        for idx, value in enumerate(case_dice.detach().cpu().flatten().tolist(), start=1):
            row[f"dice_class_{idx}"] = value
        rows.append(row)
    mean_dice = float(metric.aggregate().item())
    metric.reset()
    model.train()
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["case", "dice"] + [f"dice_class_{idx}" for idx in range(1, NUM_CLASSES)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"case": "MEAN", "dice": mean_dice})
    return mean_dice


def parse_amos_args(argv: list[str]) -> tuple[int, int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    known, remaining = parser.parse_known_args(argv)
    return known.max_train, known.max_val, remaining


def main() -> None:
    import sys

    max_train, max_val, remaining = parse_amos_args(sys.argv[1:])
    split_cases.max_train = max_train
    split_cases.max_val = max_val
    sys.argv = [sys.argv[0]] + remaining
    base.split_cases = split_cases
    base.build_transforms = build_transforms
    base.build_model = build_model
    base.evaluate = evaluate
    base.train(base.parse_args())


if __name__ == "__main__":
    main()
