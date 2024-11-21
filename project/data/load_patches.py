import re
from typing import Dict, List, TypedDict

import torch

from project.data.dirs import PATCHES_DIR


class Patch(TypedDict):
    county: str
    name: str
    resnet_features: torch.Tensor
    image_path: str
    grid_x: int
    grid_y: int


def load_patches() -> Dict[str, List[Patch]]:
    patches = []

    # Load patch image paths and features.
    for county in PATCHES_DIR.iterdir():
        for patch_path in county.iterdir():
            image_path = (
                PATCHES_DIR / county.name / patch_path.name.replace(".pt", ".png")
            )
            regex_match = re.match(r"grid(\d+)_(\d+)\.pt", patch_path.name)
            assert regex_match is not None
            grid_x, grid_y = regex_match.groups()
            patches.append(
                {
                    "county": county.name,
                    "name": patch_path.name.replace(".pt", ""),
                    "resnet_features": torch.load(patch_path),
                    "image_path": image_path,
                    "grid_x": int(grid_x),
                    "grid_y": int(grid_y),
                }
            )

    patches_per_county = {}
    for patch in patches:
        if patch["county"] not in patches_per_county:
            patches_per_county[patch["county"]] = []

        patches_per_county[patch["county"]].append(patch)

    return patches_per_county
