from pathlib import Path


ledger = Path("/root/autodl-tmp/ARM/RESULTS_LEDGER.md")
text = ledger.read_text(encoding="utf-8")
old = """| 48 train / 12 val fast probe, 16-class UNet, DiceCE | 46.02 | rejected as final; useful only for trend/protocol debugging | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_multiclass_60e` |
| 96 train / 24 val Dice+Focal resume probe | running | method-improvement probe | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_dicefocal_resume_80e` |

Status: in progress. The first AMOS22 fast probe learned but remained far below the 90.22 paper target, so a Dice+Focal resume probe with more data and stronger foreground sampling is now running before launching the next full-data training."""
new = """| 48 train / 12 val fast probe, 16-class UNet, DiceCE | 46.02 | rejected as final; useful only for trend/protocol debugging | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_multiclass_60e` |
| 96 train / 24 val Dice+Focal resume probe | 73.73 | improved but still below target | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_dicefocal_resume_80e` |
| 144 train / 36 val small-organ Tversky resume probe | running | targets low-Dice small organs | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_smallorgan_resume_80e` |

Status: in progress. Dice+Focal improved AMOS22 substantially, but per-class diagnostics show small organs remain the bottleneck, so a higher-resolution Tversky resume probe is now running before launching the next full-data training."""
if old in text:
    text = text.replace(old, new)
else:
    print("warning: expected AMOS22 table block not found")
ledger.write_text(text, encoding="utf-8")
print("updated", ledger)
