import argparse
from typing import Tuple

import torch
import finetune_spleen_monai_standard as base
from finetune_amos22_monai_standard import (
    NUM_CLASSES,
    build_transforms,
    evaluate,
    split_cases,
)
from monai.networks.nets import SwinUNETR


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        patch_size=args.swin_patch_size,
        depths=tuple(args.swin_depths),
        num_heads=tuple(args.swin_num_heads),
        window_size=tuple(args.swin_window_size),
        feature_size=args.swin_feature_size,
        norm_name="instance",
        use_checkpoint=args.swin_use_checkpoint,
        spatial_dims=3,
    )


def parse_swin_args(argv: list[str]) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
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
    split_cases.max_train = swin_args.max_train
    split_cases.max_val = swin_args.max_val or args.val_count
    for key, value in vars(swin_args).items():
        setattr(args, key, value)
    base.split_cases = split_cases
    base.build_transforms = build_transforms
    base.build_model = build_model
    base.evaluate = evaluate
    base.train(args)


if __name__ == "__main__":
    main()
