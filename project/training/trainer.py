import json
import re
from pathlib import Path
from typing import Dict, List, Literal, cast

import pandas as pd
import torch

from project.training.models import SimpleFeatureProjector

DATA_PATH = Path(__file__).parent.parent.parent / "data"
PATCHES_DIR = DATA_PATH / "patch_features/virginia"


def load_patches():
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

    patches_per_county: Dict[str, List] = {}
    for patch in patches:
        if patch["county"] not in patches_per_county:
            patches_per_county[patch["county"]] = []

        patches_per_county[patch["county"]].append(patch)

    return patches_per_county


def map_county_name_to_county_slug(county_name: str):
    if county_name == "Virginia Beach City":
        return "virginia_beach"

    elif county_name == "Chesapeake City":
        return "chesapeake"

    return county_name.replace(" ", "_").lower()


def verify_county_names():
    field_crops_df = pd.read_csv(DATA_PATH / "census_data" / "table25_parsed.csv")
    fruits_df = pd.read_csv(DATA_PATH / "census_data" / "table31_parsed.csv")

    unique_counties = set(
        [*field_crops_df["county"].unique(), *fruits_df["county"].unique()]
    )
    for county_name in unique_counties:
        if not (PATCHES_DIR / map_county_name_to_county_slug(county_name)).exists():
            print(county_name, "has no corresponding slug folder.")


def parse_field_crop_name(crop_name: str) -> tuple:
    crop_name = crop_name.replace("(See Text)", "").strip()

    parentheses_index = crop_name.rfind("(")
    units_of_measurement = crop_name[parentheses_index + 1 : -1]
    crop_name = crop_name[: parentheses_index - 1]

    if units_of_measurement == "Pounds, Shelled":
        units_of_measurement = "Pounds"

    assert units_of_measurement in [
        "Pounds",
        "Bushels",
        "Cwt",
        "Bales",
    ], units_of_measurement

    return crop_name, units_of_measurement  # type: ignore


# Instead of training from scratch, we may want to try using an autoencoder.
def unsupervised_pretraining():
    pass


def generate_crop_ontology():
    field_crops_df = pd.read_csv(DATA_PATH / "census_data" / "table25_parsed.csv")
    fruits_df = pd.read_csv(DATA_PATH / "census_data" / "table31_parsed.csv")

    # Generate an ontology.
    crop_ontology = []
    id_counter = 0

    # Identify the units of measurement for each crop.
    for crop in sorted(field_crops_df["crop"].unique()):
        original_name = crop
        crop, units_of_measurement = parse_field_crop_name(crop)
        crop_slug = re.sub(r"[^\w]+", "_", crop).lower().strip()
        crop_ontology.append(
            {"original_name": original_name, "slug": crop_slug, "id": id_counter}
        )
        id_counter += 1

    for crop in sorted(fruits_df["crop"].unique()):
        original_name = crop
        crop = crop.replace(" (See Text)", "")
        crop_slug = re.sub(r"[^\w]+", "_", crop).lower().strip()
        crop_ontology.append(
            {"original_name": original_name, "slug": crop_slug, "id": id_counter}
        )
        id_counter += 1

    return crop_ontology


def train():
    patches_per_county = load_patches()

    with open(DATA_PATH / "crop_ontology.json", "w") as f:
        json.dump(generate_crop_ontology(), f)

    # Train a CNN. For this, we will just take the mean-squared-error of each crop.
    # We will output two primary features:
    # (a) whether a crop is farmed in a county, and
    # (b) how many <units> of that crop is farmed there.
    model = SimpleFeatureProjector(512, 256, 10)

    for epoch in range(10):
        for county, patches in patches_per_county.items():
            pass


if __name__ == "__main__":
    train()
