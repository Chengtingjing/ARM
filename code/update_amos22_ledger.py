from pathlib import Path


ledger = Path("/root/autodl-tmp/ARM/RESULTS_LEDGER.md")
text = ledger.read_text(encoding="utf-8")

section = """\n## AMOS22 Multi-Class Segmentation\n\nPaper target from `REPRODUCTION_MATRIX.md`, Table `BTCV and unseen-dataset benchmark`:\n\n| Method | PT | AMOS22 Dice |\n| --- | --- | ---: |\n| ARM | 1k | 90.22 |\n\nDataset status:\n\n```text\n/root/autodl-tmp/ARM/data/amos22/amos22.zip = 22.57 GiB\n/root/autodl-tmp/ARM/data/amos22/extracted/amos22/imagesTr = 240 images\n/root/autodl-tmp/ARM/data/amos22/extracted/amos22/labelsTr = 240 labels\n/root/autodl-tmp/ARM/data/amos22/extracted/amos22/imagesVa = 120 images\n/root/autodl-tmp/ARM/data/amos22/extracted/amos22/labelsVa = 120 labels\n/root/autodl-tmp/ARM/data/amos22/extracted/amos22/imagesTs = 240 images, no local labels\n```\n\nProtocol:\n\n- Added AMOS22-specific 16-class runner at `/root/autodl-tmp/ARM/code/arm_runner/finetune_amos22_monai_standard.py`.\n- Smoke test passed at `/root/autodl-tmp/ARM/runs/smoke_monai_standard_amos22_multiclass`, confirming 16-class labels, loss, checkpointing, and sliding-window validation.\n- Full training is running at `/root/autodl-tmp/ARM/runs/monai_standard_amos22_multiclass_160e` via `/root/autodl-tmp/ARM/code/arm_runner/run_monai_standard_amos22.sh`.\n\nStatus: in progress. No accepted AMOS22 Dice yet; wait for the first full validation checkpoints before judging convergence or changing method.\n"""

start = text.find("\n## AMOS22 Multi-Class Segmentation\n")
if start != -1:
    next_start = text.find("\n## ", start + 1)
    if next_start == -1:
        text = text[:start].rstrip() + "\n" + section
    else:
        text = text[:start].rstrip() + "\n" + section + "\n" + text[next_start:].lstrip()
else:
    insert = text.find("\n## MSD Task09 Spleen\n")
    if insert == -1:
        text = text.rstrip() + "\n" + section
    else:
        text = text[:insert].rstrip() + "\n" + section + "\n" + text[insert:].lstrip()

ledger.write_text(text, encoding="utf-8")
print("updated", ledger)
