"""
For each county, extract some patches. We will place these in the `data/patches/<state>/<county>` folder,
with the name "gridX_gridY.png". Then, we will run feature extraction on these patches, with a lightweight
ResNet model trained on ImageNet. These will be stored in `data/patch_features/<state>/<county>`.
"""

import os
from pathlib import Path

import numpy as np
import PIL.Image
import tqdm

DATA_PATH = Path(os.path.dirname(__file__)) / "../../data"


def extract_patches(state: str, county: str, max_empty_pixel_ratio: float):
    satellite_imagery_path = DATA_PATH / "satellite_images" / state / county
    patches_path = DATA_PATH / "patches" / state / county

    if not patches_path.exists():
        patches_path.mkdir(parents=True)

    # Load the RGB image for the state.
    image = np.array(PIL.Image.open(satellite_imagery_path / "render_rgb.png"))

    # Create 224x224 patches.
    grid_width = image.shape[1] // 224
    grid_height = image.shape[0] // 224

    # Identify a patch.
    for grid_x in range(grid_width):
        for grid_y in range(grid_height):
            patch = image[
                grid_y * 224 : (grid_y + 1) * 224, grid_x * 224 : (grid_x + 1) * 224
            ]

            empty_pixels = np.all(patch == 0, axis=-1).sum()
            if (empty_pixels / (224 * 224)) > max_empty_pixel_ratio:
                continue

            PIL.Image.fromarray(patch).save(patches_path / f"grid{grid_x}_{grid_y}.png")


def main():
    county_dirs = [
        c for c in (DATA_PATH / "satellite_images" / "virginia").iterdir() if c.is_dir()
    ]
    for county in tqdm.tqdm(county_dirs, desc="Extracting patches"):
        state = county.parent.name
        county = county.name
        extract_patches(state, county, max_empty_pixel_ratio=0.2)


if __name__ == "__main__":
    main()
