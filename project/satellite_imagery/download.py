import os
from datetime import datetime

import ee
import geemap
import numpy as np
import PIL.Image
import tifffile

from .geojson import load_counties_by_state
from .merge_rgb import merge_rgb


def download_bands(
    bbox: list[float],
    output_dir: str,
    collection_name: str,
    band_names: list[str],
    scale: float,
    start_date: str,
    end_date: str,
    *,
    cloud_cover_property: str | None = "CLOUDY_PIXEL_PERCENTAGE",
    cloud_cover_threshold: float = 0.2,
) -> dict[str, str]:
    """
    Download satellite imagery from Google Earth Engine for a specified bounding box.

    Parameters:
    bbox (list): List of coordinates [min_lon, min_lat, max_lon, max_lat]
    output_dir (str): Directory to save the downloaded images
    collection_name (str): Name of the Earth Engine collection to use
    start_date (str): Start date in 'YYYY-MM-DD' format (default: 30 days ago)
    end_date (str): End date in 'YYYY-MM-DD' format (default: today)
    cloud_cover_threshold (float): Maximum cloud cover percentage (0-100)
    export_separate_bands (bool): Whether to export each band separately
    """
    # Initialize Earth Engine
    ee.Initialize()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create geometry from bounding box
    geometry = ee.geometry.Geometry.Rectangle(bbox)

    # Get image collection
    collection = (
        ee.imagecollection.ImageCollection(collection_name)
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
    )
    if cloud_cover_property is not None:
        collection = collection.filter(
            ee.filter.Filter.lt(cloud_cover_property, cloud_cover_threshold)
        )
        # # Get the least cloudy image
        # image: ee.image.Image = collection.sort(cloud_cover_property).first()

    image: ee.image.Image = collection.first()

    if image is None:
        raise ValueError("No images found for the specified criteria.")

    filenames: dict[str, str] = {}

    for band_name in band_names:
        band_image = image.select(band_name)
        filename = os.path.join(
            output_dir,
            f"{collection_name.replace('/', '_')}_{start_date}_to_{end_date}.tif",
        )
        # For some reason, they modify the filename.
        filenames[band_name] = f"{filename[:-4]}.{band_name}.tif"
        geemap.ee_export_image(
            band_image,
            filename=filename,
            scale=scale,
            region=geometry,
            file_per_band=True,
        )

    return filenames


def download_region(bbox: list[float], output_dir: str):
    paths = {}

    # RGB.
    result = download_bands(
        bbox,
        output_dir,
        "LANDSAT/LC09/C02/T1",
        band_names=["B4", "B3", "B2"],
        scale=30,
        cloud_cover_property="CLOUD_COVER",
        cloud_cover_threshold=0.2,
        start_date="2015-01-01",
        end_date=datetime.strftime(datetime.now(), "%Y-%m-%d"),
    )
    paths["red"] = result["B4"]
    paths["green"] = result["B3"]
    paths["blue"] = result["B2"]

    # Now, we can create a combined RGB image.
    merge_rgb(
        paths["red"],
        paths["green"],
        paths["blue"],
        sensor_max=30000,
    ).save(os.path.join(output_dir, "render_rgb.png"))

    # Elevation.
    result = download_bands(
        bbox,
        # [bbox[0], bbox[1], bbox[0] + 0.01, bbox[1] + 0.01],
        output_dir,
        "USGS/3DEP/1m",
        band_names=["elevation"],
        scale=30,
        cloud_cover_property=None,
        # Dataset has old date range.
        start_date="2006-01-01",
        end_date="2015-01-01",
    )
    paths["elevation"] = result["elevation"]

    # Convert this to .png.
    paths["elevation_png"] = os.path.join(output_dir, "render_elevation.png")

    # NOTE that the elevation is stored as floating-point values.
    # Therefore, the png file is just for us to inspect.
    with tifffile.TiffFile(paths["elevation"]) as tif:
        elevation_data = tif.asarray()

    elevation_data_scaled = elevation_data - elevation_data.min()
    elevation_data_scaled = elevation_data_scaled / elevation_data_scaled.max()

    PIL.Image.fromarray((255 * elevation_data_scaled).astype(np.uint8)).save(
        paths["elevation_png"]
    )

    # Lithography.
    # result = download_bands(
    #     bbox,
    #     output_dir,
    #     "CSP/ERGo/1_0/US/lithology",
    #     band_names=["b1"],
    #     scale=90,
    #     cloud_cover_property=None,
    # )

    return paths


def download_counties(state: str):
    counties = load_counties_by_state()[state]

    for county in counties:
        print(county)

        if county["name"] == "Fairfax":
            county_name_slug = county["name"].replace(" ", "_").lower()
            state_name_slug = state.replace(" ", "_").lower()

            paths = download_region(
                county["bbox"],
                f"data/satellite_images/{state_name_slug}/{county_name_slug}",
            )

            print(paths)

            break


if __name__ == "__main__":
    download_counties("Virginia")
