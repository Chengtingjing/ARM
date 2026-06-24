from pathlib import Path


ledger = Path("/root/autodl-tmp/ARM/RESULTS_LEDGER.md")
text = ledger.read_text(encoding="utf-8")
needle = "Status: in progress. No accepted AMOS22 Dice yet; wait for the first full validation checkpoints before judging convergence or changing method."
replacement = """Current local evidence:\n\n| Protocol | Best Dice | Status | Path |\n| --- | ---: | --- | --- |\n| 48 train / 12 val fast probe, 16-class UNet, DiceCE | 46.02 | rejected as final; useful only for trend/protocol debugging | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_multiclass_60e` |\n| 96 train / 24 val Dice+Focal resume probe | running | method-improvement probe | `/root/autodl-tmp/ARM/runs/probe_monai_standard_amos22_dicefocal_resume_80e` |\n\nStatus: in progress. The first AMOS22 fast probe learned but remained far below the 90.22 paper target, so a Dice+Focal resume probe with more data and stronger foreground sampling is now running before launching the next full-data training."""
if needle in text:
    text = text.replace(needle, replacement)
else:
    marker = "## AMOS22 Multi-Class Segmentation"
    print("warning: marker not updated; AMOS section may already differ")
ledger.write_text(text, encoding="utf-8")
print("updated", ledger)
