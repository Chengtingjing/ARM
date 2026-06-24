import json
from pathlib import Path

import nibabel as nib
import numpy as np


root = Path("/root/autodl-tmp/ARM/data/amos22/extracted/amos22")
dataset = json.load((root / "dataset.json").open(encoding="utf-8"))
print("dataset_keys", sorted(dataset.keys()))
print("modality", dataset.get("modality"))
print("labels", dataset.get("labels"))
print("numTraining", dataset.get("numTraining"), "numTest", dataset.get("numTest"))
for folder in ["imagesTr", "labelsTr", "imagesVa", "labelsVa", "imagesTs", "labelsTs"]:
    path = root / folder
    print(folder, len(list(path.glob("*.nii.gz"))) if path.exists() else 0)
for folder in ["labelsTr", "labelsVa"]:
    for p in sorted((root / folder).glob("*.nii.gz"))[:3]:
        arr = np.asanyarray(nib.load(str(p)).dataobj)
        print(folder, p.name, arr.shape, sorted(np.unique(arr).astype(int).tolist())[:50])
