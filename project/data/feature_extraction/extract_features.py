"""
We iterate over all .png files in the data/patches folder. Then, we apply a ResNet model to them to extract featuers, which will be stored as Pytorch files.
"""

import os
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T
import tqdm
from torchvision.models.resnet import ResNet, ResNet18_Weights, resnet18

DATA_PATH = Path(os.path.dirname(__file__)) / "../../data"
INFERENCE_TRANSFORMS = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_features(model: ResNet, state: str, county: str):
    county_out_dir = DATA_PATH / "patch_features" / state / county

    if not county_out_dir.exists():
        county_out_dir.mkdir(parents=True)

    patch_files = list((DATA_PATH / "patches" / state / county).glob("*.png"))
    patch_tensor_list: list[torch.Tensor] = []
    patch_tensor_filenames: list[str] = []

    if len(patch_files) == 0:
        return

    for patch_file in patch_files:
        image = np.array(PIL.Image.open(patch_file))
        patch: torch.Tensor = INFERENCE_TRANSFORMS(image)  # type: ignore
        patch_tensor_list.append(patch)
        patch_tensor_filenames.append(patch_file.name.replace(".png", ".pt"))

    patches_tensor = torch.stack(patch_tensor_list)

    with torch.no_grad():
        features = model(patches_tensor)

    for i, filename in enumerate(patch_tensor_filenames):
        torch.save(features[i], county_out_dir / filename)


def main():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # Cut off the last layer.
    model.fc = torch.nn.Identity()  # type: ignore

    counties = list((DATA_PATH / "patches" / "virginia").iterdir())

    for county in tqdm.tqdm(counties, desc="Extracting features"):
        state = county.parent
        extract_features(model, state.name, county.name)


if __name__ == "__main__":
    main()
