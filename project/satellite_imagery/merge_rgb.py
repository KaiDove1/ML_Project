import PIL.Image
import numpy as np


def merge_rgb(
    r_tiff_path: str, g_tiff_path: str, b_tiff_path: str, sensor_max: float = 20000
) -> PIL.Image.Image:
    R = PIL.Image.open(r_tiff_path)
    G = PIL.Image.open(g_tiff_path)
    B = PIL.Image.open(b_tiff_path)

    data = np.stack([np.array(R), np.array(G), np.array(B)], axis=-1)

    # Scale to normal bounds.
    data = np.minimum(data / sensor_max, 1)

    # Convert to 8-bit.
    data = (data * 255).astype(np.uint8)

    # Create an image from the array.
    image = PIL.Image.fromarray(data)

    return image
