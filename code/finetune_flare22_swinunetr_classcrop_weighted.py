import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import torch
import finetune_spleen_monai_standard as base
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
)


def _case_id_from_image(path: Path) -> str:
    name = path.name
    if name.endswith("_0000.nii.gz"):
        return name[: -len("_0000.nii.gz")]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _case_id_from_label(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _collect_pairs(image_dir: Path, label_dir: Path) -> List[dict]:
    images = sorted(p for p in image_dir.rglob("*.nii.gz") if not p.name.startswith("._"))
    labels = sorted(p for p in label_dir.rglob("*.nii.gz") if not p.name.startswith("._"))
    label_map = {_case_id_from_label(p): p for p in labels}
    items = []
    for image in images:
        case_id = _case_id_from_image(image)
        label = label_map.get(case_id)
        if label is not None:
            items.append({"image": str(image), "label": str(label), "case": case_id})
    return items


def split_cases(data_root: str, val_count: int, seed: int) -> Tuple[List[dict], List[dict]]:
    root = Path(data_root)
    items = []
    items.extend(
        _collect_pairs(
            root / "Training" / "FLARE22_LabeledCase50" / "images",
            root / "Training" / "FLARE22_LabeledCase50" / "labels",
        )
    )
    items.extend(_collect_pairs(root / "Tuning" / "images", root / "Tuning" / "labels"))
    if not items:
        raise RuntimeError(f"No FLARE22 image/label pairs found under {root}")
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = min(max(val_count, 1), len(items) - 1)
    return items[val_count:], items[:val_count]


def build_transforms(args: argparse.Namespace) -> Tuple[Compose, Compose]:
    pixdim = tuple(args.pixdim)
    roi_size = tuple(args.roi_size)
    base_steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
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
        base_steps
        + [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                num_samples=args.num_samples,
                num_classes=args.num_classes,
                ratios=[0.05, 0.5, 0.5, 0.5, 1.0, 0.75, 0.75, 6.0, 6.0, 0.75, 6.0, 0.75, 1.0, 0.5],
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
    val = Compose(base_steps + [EnsureTyped(keys=["image", "label"])])
    return train, val


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return SwinUNETR(
        in_channels=1,
        out_channels=args.num_classes,
        patch_size=args.swin_patch_size,
        depths=tuple(args.swin_depths),
        num_heads=tuple(args.swin_num_heads),
        window_size=tuple(args.swin_window_size),
        feature_size=args.swin_feature_size,
        norm_name="instance",
        use_checkpoint=args.swin_use_checkpoint,
        spatial_dims=3,
    )


class ClassWeightedDiceFocalLoss(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        weights = [0.05] + [1.0] * (args.num_classes - 1)
        for label in (7, 8, 10):
            if label < len(weights):
                weights[label] = 8.0
        for label in (4, 6, 12):
            if label < len(weights):
                weights[label] = 2.0
        self.register_buffer("class_weight", torch.tensor(weights, dtype=torch.float32, device=device))
        self.lambda_dice = args.lambda_dice
        self.lambda_focal = args.lambda_focal
        self.gamma = args.focal_gamma

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        target = label[:, 0].long()
        log_prob = torch.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(-50.0)
        pt = log_pt.exp()
        focal = -self.class_weight[target] * torch.pow(1.0 - pt, self.gamma) * log_pt
        return self.lambda_dice * self.dice(logits, label) + self.lambda_focal * focal.mean()


def build_loss(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return ClassWeightedDiceFocalLoss(args, device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_csv: Path | None = None,
) -> float:
    model.eval()
    post = Compose(
        [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True, to_onehot=args.num_classes),
            AsDiscreted(keys="label", to_onehot=args.num_classes),
        ]
    )
    metric = DiceMetric(include_background=False, reduction="mean")
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
        case_metric = DiceMetric(include_background=False, reduction="mean")(y_pred=preds, y=labels)
        if case_metric.numel() > 1:
            case_metric = torch.nanmean(case_metric)
        case_name = batch.get("case", ["unknown"])[0]
        rows.append({"case": case_name, "dice": float(case_metric.item())})
    mean_dice = float(metric.aggregate().item())
    metric.reset()
    model.train()
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "dice"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"case": "MEAN", "dice": mean_dice})
    return mean_dice


def parse_swin_args(argv: list[str]) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--swin-patch-size", type=int, default=2)
    parser.add_argument("--swin-depths", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--swin-num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--swin-window-size", type=int, nargs=3, default=[7, 7, 7])
    parser.add_argument("--swin-feature-size", type=int, default=24)
    parser.add_argument("--swin-use-checkpoint", action="store_true")
    return parser.parse_known_args(argv)


def main() -> None:
    import sys

    swin_args, remaining = parse_swin_args(sys.argv[1:])
    args = base.parse_args(remaining)
    for key, value in vars(swin_args).items():
        setattr(args, key, value)
    base.split_cases = split_cases
    base.build_transforms = build_transforms
    base.build_model = build_model
    base.build_loss = build_loss
    base.evaluate = evaluate
    base.train(args)


if __name__ == "__main__":
    main()
