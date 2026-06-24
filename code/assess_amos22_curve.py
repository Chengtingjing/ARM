import csv
from pathlib import Path

RUN = Path('/root/autodl-tmp/ARM/runs/swinunetr_amos22_full_dicefocal_180e_v4')
TARGET = 90.22
LOWER = TARGET - 3.0
MIN_RECENT_GAIN_PP = 3.0


def read_points(path: Path):
    points = []
    if not path.exists():
        return points
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                val = float(row.get('val_dice', -1))
            except Exception:
                continue
            if val >= 0:
                points.append((int(row['epoch']), val * 100.0))
    return points


def main():
    points = read_points(RUN / 'metrics.csv')
    print(f'run={RUN}')
    print(f'target={TARGET:.2f} lower_bound={LOWER:.2f}')
    if not points:
        print('status=no_validation_points')
        return
    print('validation_points=')
    for epoch, dice in points:
        print(f'  epoch={epoch} dice={dice:.2f} gap_to_lower={dice - LOWER:+.2f} pp')
    best_epoch, best_dice = max(points, key=lambda item: item[1])
    print(f'best_epoch={best_epoch} best_dice={best_dice:.2f}')
    if len(points) >= 2:
        prev_epoch, prev_dice = points[-2]
        last_epoch, last_dice = points[-1]
        gain = last_dice - prev_dice
        print(f'recent_gain=epoch{prev_epoch}_to_epoch{last_epoch}:{gain:+.2f} pp')
        if last_dice >= LOWER:
            advice = 'accepted_or_ready_for_variant_scan'
        elif gain < MIN_RECENT_GAIN_PP and last_epoch >= 25:
            advice = 'switch_to_v6_smallorgan_tversky_resume_when_current_run_exits'
        else:
            advice = 'continue_current_run_until_next_validation'
    else:
        advice = 'continue_current_run_until_more_validation_points'
    print(f'advice={advice}')


if __name__ == '__main__':
    main()
