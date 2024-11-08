"""
Example file showing how bands can be combined to create an RGB image.
"""

import PIL.Image
import numpy as np


def main():
    # R = PIL.Image.open(
    #     "data/satellite_images/california/san_francisco/LANDSAT_LC09_C02_T1_L2_2024-01-01_to_2024-01-31_SR_B4.SR_B4.tif"
    # )
    # G = PIL.Image.open(
    #     "data/satellite_images/california/san_francisco/LANDSAT_LC09_C02_T1_L2_2024-01-01_to_2024-01-31_SR_B3.SR_B3.tif"
    # )
    # B = PIL.Image.open(
    #     "data/satellite_images/california/san_francisco/LANDSAT_LC09_C02_T1_L2_2024-01-01_to_2024-01-31_SR_B2.SR_B2.tif"
    # )

    PIL.Image.open(
        "data/satellite_images/virginia/fairfax/USGS_3DEP_1m_2006-01-01_to_2015-01-01_elevation.elevation.tif"
    ).show()

    R = PIL.Image.open(
        "data/satellite_images/virginia/fairfax/LANDSAT_LC09_C02_T1_L2_2023-11-01_to_2024-10-31_SR_B4.SR_B4.tif"
    )
    G = PIL.Image.open(
        "data/satellite_images/virginia/fairfax/LANDSAT_LC09_C02_T1_L2_2023-11-01_to_2024-10-31_SR_B3.SR_B3.tif"
    )
    B = PIL.Image.open(
        "data/satellite_images/virginia/fairfax/LANDSAT_LC09_C02_T1_L2_2023-11-01_to_2024-10-31_SR_B2.SR_B2.tif"
    )

    data = np.stack([np.array(R), np.array(G), np.array(B)], axis=-1)

    print(data.dtype)
    print(data.min(), data.max())
    print(data.shape)

    # Scale to normal bounds.
    data = np.minimum(data / 20000, 1)  # data.max()

    # Convert to 8-bit.
    data = (data * 255).astype(np.uint8)

    # Create an image from the array.
    image = PIL.Image.fromarray(data)

    # Show the image.
    image.show()


if __name__ == "__main__":
    main()
