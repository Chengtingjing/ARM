import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscreted, Compose, KeepLargestConnectedComponentd

import finetune_amos22_monai_standard as amos
import finetune_amos22_swinunetr as amos_swin


NUM_CLASSES = amos.NUM_CLASSES


def _namespace_from_json(path: Path) -> argparse.Namespace:
    data = json.loads(path.read_text(encoding='utf-8'))
    return argparse.Namespace(**data)


def _apply_missing_arg_defaults(args: argparse.Namespace) -> None:
    defaults = {
        'cache_rate': 1.0,
        'num_workers': 2,
        'sw_batch_size': 1,
        'overlap': 0.625,
        'keep_largest': False,
        'max_train': 0,
        'max_val': 0,
        'swin_patch_size': 2,
        'swin_depths': [2, 2, 2, 2],
        'swin_num_heads': [3, 6, 12, 24],
        'swin_window_size': [7, 7, 7],
        'swin_feature_size': 24,
        'swin_use_checkpoint': False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def _is_swin_args(args: argparse.Namespace, checkpoint_path: Path) -> bool:
    if hasattr(args, 'swin_feature_size') and hasattr(args, 'swin_depths'):
        # Older UNet args may receive defaults above; use run/checkpoint naming as an extra signal.
        return 'swinunetr' in str(checkpoint_path).lower() or 'swin' in str(getattr(args, 'output_dir', '')).lower()
    return False


def _build_model(args: argparse.Namespace, checkpoint_path: Path) -> torch.nn.Module:
    if _is_swin_args(args, checkpoint_path):
        return amos_swin.build_model(args)
    return amos.build_model(args)


@torch.no_grad()
def evaluate_variant(model, loader, device, args, keep_largest: bool, overlap: float):
    model.eval()
    post_steps = [
        AsDiscreted(keys='pred', argmax=True, to_onehot=NUM_CLASSES),
        AsDiscreted(keys='label', to_onehot=NUM_CLASSES),
    ]
    if keep_largest:
        post_steps.append(
            KeepLargestConnectedComponentd(
                keys='pred',
                applied_labels=list(range(1, NUM_CLASSES)),
                is_onehot=True,
                independent=True,
            )
        )
    post = Compose(post_steps)
    metric = DiceMetric(include_background=False, reduction='mean', ignore_empty=True)
    rows = []
    for batch in loader:
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        logits = sliding_window_inference(
            image,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode='gaussian',
        )
        data = [{'pred': p, 'label': l} for p, l in zip(decollate_batch(logits), decollate_batch(label))]
        data = [post(d) for d in data]
        preds = torch.stack([d['pred'] for d in data])
        labels = torch.stack([d['label'] for d in data])
        metric(y_pred=preds, y=labels)
        case_dice = DiceMetric(include_background=False, reduction='mean_batch', ignore_empty=True)(
            y_pred=preds,
            y=labels,
        )
        row = {'case': batch.get('case', ['unknown'])[0], 'dice': float(torch.nanmean(case_dice).item())}
        for idx, value in enumerate(case_dice.detach().cpu().flatten().tolist(), start=1):
            row[f'dice_class_{idx}'] = value
        rows.append(row)
    mean_dice = float(metric.aggregate().item())
    metric.reset()
    return mean_dice, rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate AMOS22 checkpoint post-processing variants')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', default='checkpoint_best.pt')
    parser.add_argument('--output-csv', default='')
    parser.add_argument('--overlaps', type=float, nargs='+', default=[0.5, 0.625, 0.75])
    parser.add_argument('--keep-largest-values', nargs='+', default=['true', 'false'])
    parser.add_argument('--cache-rate', type=float, default=-1.0)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--max-val', type=int, default=0, help='Optional cap for quick diagnostic scans only')
    parser.add_argument('--cpu', action='store_true')
    cli = parser.parse_args()

    run_dir = Path(cli.run_dir)
    args = _namespace_from_json(run_dir / 'args.json')
    _apply_missing_arg_defaults(args)
    amos.split_cases.max_train = 0
    amos.split_cases.max_val = cli.max_val
    _, val_items = amos.split_cases(args.data_root, getattr(args, 'val_count', 1), getattr(args, 'seed', 42))
    _, val_tf = amos.build_transforms(args)
    cache_rate = args.cache_rate if cli.cache_rate < 0 else cli.cache_rate
    num_workers = args.num_workers if cli.num_workers < 0 else cli.num_workers
    val_ds = CacheDataset(val_items, val_tf, cache_rate=cache_rate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    checkpoint_path = run_dir / cli.checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() and not cli.cpu else 'cpu')
    model = _build_model(args, checkpoint_path).to(device)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    keep_values = [x.lower() in {'1', 'true', 'yes', 'y'} for x in cli.keep_largest_values]
    summary = []
    best = SimpleNamespace(dice=-1.0, rows=None, config=None)
    for keep_largest in keep_values:
        for overlap in cli.overlaps:
            dice, rows = evaluate_variant(model, val_loader, device, args, keep_largest, overlap)
            record = {
                'keep_largest': keep_largest,
                'overlap': overlap,
                'mean_dice': dice,
            }
            summary.append(record)
            print(record, flush=True)
            if dice > best.dice:
                best = SimpleNamespace(dice=dice, rows=rows, config=record)

    output_csv = Path(cli.output_csv) if cli.output_csv else run_dir / 'amos22_eval_variant_summary.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['keep_largest', 'overlap', 'mean_dice'])
        writer.writeheader()
        writer.writerows(summary)

    best_csv = output_csv.with_name(output_csv.stem + '_best_cases.csv')
    fieldnames = ['case', 'dice'] + [f'dice_class_{idx}' for idx in range(1, NUM_CLASSES)]
    with best_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best.rows or [])
        writer.writerow({'case': 'MEAN', 'dice': best.dice})
    print(f'best={best.config}', flush=True)
    print(f'summary_csv={output_csv}', flush=True)
    print(f'best_cases_csv={best_csv}', flush=True)


if __name__ == '__main__':
    main()
