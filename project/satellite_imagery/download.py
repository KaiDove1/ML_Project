import ee
import geemap
import os
from datetime import datetime, timedelta
from typing import Optional
from project.satellite_imagery.geojson import load_counties_by_state


def download_bands(
    bbox: list[float],
    output_dir: str,
    collection_name: str,
    band_names: list[str],
    scale: float,
    *,
    cloud_cover_property: str | None = "CLOUDY_PIXEL_PERCENTAGE",
    cloud_cover_threshold: float = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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

    # Set default dates if not provided
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

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

        # Get the least cloudy image
        image: ee.image.Image = collection.sort(cloud_cover_property).first()
    else:
        image: ee.image.Image = collection.first()

    if image is None:
        raise ValueError("No images found for the specified criteria.")

    filenames: dict[str, str] = {}

    for band_name in band_names:
        band_image = image.select(band_name)
        filename = os.path.join(
            output_dir,
            f"{collection_name.replace('/', '_')}_{start_date}_to_{end_date}_{band_name}.tif",
        )
        filenames[band_name] = filename
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
        "LANDSAT/LC09/C02/T1_L2",
        band_names=["SR_B4", "SR_B3", "SR_B2"],
        scale=30,
        cloud_cover_property="CLOUD_COVER",
    )
    paths["red"] = result["SR_B4"]
    paths["green"] = result["SR_B3"]
    paths["blue"] = result["SR_B2"]

    # Elevation.
    result = download_bands(
        bbox,
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
