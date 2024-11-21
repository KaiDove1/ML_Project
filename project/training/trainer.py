import json
import re
from pathlib import Path
from typing import Dict, List, Literal, cast

import pandas as pd
import torch

from project.training.models import SimpleFeatureProjector

DATA_PATH = Path(__file__).parent.parent.parent / "data"
PATCHES_DIR = DATA_PATH / "patch_features/virginia"


# Instead of training from scratch, we may want to try using an autoencoder.
def unsupervised_pretraining():
    pass


def train():
    patches_per_county = load_patches()

    # with open(DATA_PATH / "crop_ontology.json", "w") as f:
    #     json.dump(generate_crop_ontology(), f)

    with open(DATA_PATH / "crop_ontology.json", "r") as f:
        crop_ontology = json.load(f)

    # Train a CNN. For this, we will just take the mean-squared-error of each crop.
    # We will output two primary features:
    # (a) whether a crop is farmed in a county, and
    # (b) how many <units> of that crop is farmed there.
    # We have 60 classes.
    model = SimpleFeatureProjector(512, 256, 60)

    for epoch in range(10):
        for county, patches in patches_per_county.items():
            # Get crop counts for this county.

            patch_features = torch.stack(
                [patch["resnet_features"] for patch in patches]
            )
            print(patch_features.shape)
            result = model(patch_features)
            print(result)
            break

        break


if __name__ == "__main__":
    train()
