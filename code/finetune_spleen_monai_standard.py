import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    Lambdad,
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

from train_arm_mim import seed_everything


def split_cases(data_root: str, val_count: int, seed: int) -> Tuple[List[dict], List[dict]]:
    root = Path(data_root)
    images = sorted(p for p in (root / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
    labels = sorted(p for p in (root / "labelsTr").glob("*.nii.gz") if not p.name.startswith("._"))
    label_map = {p.name: p for p in labels}
    items = [{"image": str(p), "label": str(label_map[p.name]), "case": p.name} for p in images if p.name in label_map]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[val_count:], items[:val_count]


def build_transforms(args: argparse.Namespace) -> Tuple[Compose, Compose]:
    pixdim = tuple(args.pixdim)
    roi_size = tuple(args.roi_size)
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["label"], func=lambda x: (x > 0).to(x.dtype)),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ]
    train = Compose(
        base
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
    val = Compose(base + [EnsureTyped(keys=["image", "label"])])
    return train, val


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=tuple(args.channels),
        strides=tuple(args.strides),
        num_res_units=args.num_res_units,
        norm="INSTANCE",
    )


class TverskyCELoss(torch.nn.Module):
    def __init__(
        self,
        tversky_alpha: float,
        tversky_beta: float,
        lambda_tversky: float,
        lambda_ce: float,
        ce_weight: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
        self.ce = torch.nn.CrossEntropyLoss(weight=ce_weight)
        self.lambda_tversky = lambda_tversky
        self.lambda_ce = lambda_ce

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.lambda_tversky * self.tversky(logits, label) + self.lambda_ce * self.ce(
            logits,
            label[:, 0].long(),
        )


def build_loss(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    ce_weight = None
    if args.ce_weight_bg > 0 and args.ce_weight_fg > 0:
        ce_weight = torch.tensor([args.ce_weight_bg, args.ce_weight_fg], dtype=torch.float32, device=device)
    if args.loss == "dice_ce":
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=ce_weight,
            lambda_dice=args.lambda_dice,
            lambda_ce=args.lambda_ce,
        )
    if args.loss == "dice_focal":
        return DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=ce_weight,
            gamma=args.focal_gamma,
            lambda_dice=args.lambda_dice,
            lambda_focal=args.lambda_focal,
        )
    if args.loss == "tversky_ce":
        return TverskyCELoss(
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            lambda_tversky=args.lambda_tversky,
            lambda_ce=args.lambda_ce,
            ce_weight=ce_weight,
        )
    raise ValueError(f"Unsupported loss: {args.loss}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_csv: Path | None = None,
) -> float:
    model.eval()
    post_steps = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
    ]
    if args.keep_largest:
        post_steps.append(
            KeepLargestConnectedComponentd(
                keys="pred",
                applied_labels=[1],
                is_onehot=True,
                independent=False,
            )
        )
    post = Compose(post_steps)
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
        case_dice = DiceMetric(include_background=False, reduction="mean_batch")(y_pred=preds, y=labels)
        case_name = batch.get("case", ["unknown"])[0]
        rows.append({"case": case_name, "dice": float(case_dice.item())})
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


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.sharing_strategy:
        torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    train_items, val_items = split_cases(args.data_root, args.val_count, args.seed)
    train_tf, val_tf = build_transforms(args)
    train_ds = CacheDataset(train_items, train_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_ds = CacheDataset(val_items, val_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)

    model = build_model(args).to(device)
    loss_fn = build_loss(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if args.auto_resume and resume_path is None:
        candidate = out_dir / "checkpoint_last.pt"
        if candidate.exists():
            resume_path = candidate

    best = -1.0
    step = 0
    start_epoch = 1
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        required = ["model", "optimizer", "scheduler", "epoch", "best_dice", "step"]
        missing = [key for key in required if key not in ckpt]
        if missing:
            raise RuntimeError(
                f"resume checkpoint {resume_path} is not a full training checkpoint; "
                f"missing keys: {missing}"
            )
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best = float(ckpt["best_dice"])
        step = int(ckpt["step"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(
            f"resumed_training from={resume_path} start_epoch={start_epoch} "
            f"step={step} best={best:.6f}",
            flush=True,
        )
    elif args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu")
        state = ckpt["model"]
        if args.init_encoder_only:
            current = model.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key.startswith(args.init_encoder_prefix) and key in current and current[key].shape == value.shape
            }
            current.update(compatible)
            model.load_state_dict(current)
            print(
                f"loaded_encoder_keys={len(compatible)} prefix={args.init_encoder_prefix} "
                f"from={args.init_checkpoint}",
                flush=True,
            )
        else:
            model.load_state_dict(state)

    log_path = out_dir / "metrics.csv"
    append_metrics = start_epoch > 1 and log_path.exists()
    with log_path.open("a" if append_metrics else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_dice", "best_dice", "lr"])
        if not append_metrics:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for batch in train_loader:
                step += 1
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(image)
                    loss = loss_fn(logits, label)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                total += float(loss.detach().item()) * image.shape[0]
                count += image.shape[0]
            scheduler.step()
            if epoch % args.eval_every == 0 or epoch == 1:
                val_dice = evaluate(model, val_loader, device, args)
            else:
                val_dice = -1.0
            if val_dice > best:
                best = val_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "epoch": epoch,
                        "step": step,
                        "val_dice": val_dice,
                        "best_dice": best,
                        "args": vars(args),
                    },
                    out_dir / "checkpoint_best.pt",
                )
                evaluate(model, val_loader, device, args, out_dir / "full_volume_best.csv")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "epoch": epoch,
                    "step": step,
                    "val_dice": val_dice,
                    "best_dice": best,
                    "args": vars(args),
                },
                out_dir / "checkpoint_last.pt",
            )
            row = {
                "epoch": epoch,
                "step": step,
                "train_loss": total / max(count, 1),
                "val_dice": val_dice,
                "best_dice": best,
                "lr": opt.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(
                f"epoch={epoch} step={step} loss={row['train_loss']:.5f} "
                f"val_dice={val_dice:.4f} best={best:.4f}",
                flush=True,
            )


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MONAI standard preprocessing Spleen segmentation")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--val-count", type=int, default=8)
    p.add_argument("--roi-size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--pixdim", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    p.add_argument("--a-min", type=float, default=-175.0)
    p.add_argument("--a-max", type=float, default=250.0)
    p.add_argument("--pos-ratio", type=float, default=1.0)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--keep-largest", dest="keep_largest", action="store_true")
    p.add_argument("--no-keep-largest", dest="keep_largest", action="store_false")
    p.set_defaults(keep_largest=True)
    p.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 2])
    p.add_argument("--num-res-units", type=int, default=2)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--init-encoder-only", action="store_true")
    p.add_argument("--init-encoder-prefix", default="swinViT")
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", choices=["dice_ce", "dice_focal", "tversky_ce"], default="dice_ce")
    p.add_argument("--lambda-dice", type=float, default=1.0)
    p.add_argument("--lambda-ce", type=float, default=1.0)
    p.add_argument("--lambda-focal", type=float, default=1.0)
    p.add_argument("--lambda-tversky", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--tversky-alpha", type=float, default=0.3)
    p.add_argument("--tversky-beta", type=float, default=0.7)
    p.add_argument("--ce-weight-bg", type=float, default=0.0)
    p.add_argument("--ce-weight-fg", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=12.0)
    p.add_argument("--sharing-strategy", choices=["file_descriptor", "file_system"], default="")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
