import ee
import geemap
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def download_ee_imagery(
    bbox: List[float],
    output_dir: str,
    collection_name: str = "COPERNICUS/S2_SR",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cloud_cover_threshold: float = 20,
    export_separate_bands: bool = True,
) -> None:
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
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Create geometry from bounding box
    geometry = ee.geometry.Geometry.Rectangle(bbox)

    # Collection configurations
    COLLECTIONS = {
        "COPERNICUS/S2_SR": {
            "rgb_bands": ["B4", "B3", "B2"],
            "all_bands": [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B11",
                "B12",
            ],
            "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
            "scale": 10,  # 10 meters per pixel for most bands
            "vis_params": {"min": 0, "max": 3000},
        },
        "LANDSAT/LC09/C02/T1_L2": {
            "rgb_bands": ["SR_B4", "SR_B3", "SR_B2"],
            "all_bands": [
                "SR_B1",
                "SR_B2",
                "SR_B3",
                "SR_B4",
                "SR_B5",
                "SR_B6",
                "SR_B7",
            ],
            "cloud_property": "CLOUD_COVER",
            "scale": 30,  # 30 meters per pixel
            "vis_params": {"min": 0, "max": 30000},
        },
        "MODIS/006/MOD09GA": {
            "rgb_bands": ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
            "all_bands": [
                "sur_refl_b01",
                "sur_refl_b02",
                "sur_refl_b03",
                "sur_refl_b04",
                "sur_refl_b05",
                "sur_refl_b06",
                "sur_refl_b07",
            ],
            "cloud_property": "state_1km",  # MODIS uses a different cloud masking approach
            "scale": 500,  # 500 meters per pixel
            "vis_params": {"min": -100, "max": 16000},
        },
    }

    if collection_name not in COLLECTIONS:
        raise ValueError(
            f"Unsupported collection. Supported collections are: {list(COLLECTIONS.keys())}"
        )

    collection_config = COLLECTIONS[collection_name]

    # Get image collection
    collection = (
        ee.imagecollection.ImageCollection(collection_name)
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
    )

    # Apply cloud filtering if applicable
    if collection_config["cloud_property"] == "state_1km":
        # Special handling for MODIS cloud masking
        collection = collection.filter(
            ee.filter.Filter.lt("state_1km", cloud_cover_threshold)
        )
    else:
        collection = collection.filter(
            ee.filter.Filter.lt(
                collection_config["cloud_property"], cloud_cover_threshold
            )
        )

    # Get the least cloudy image
    image: ee.image.Image = collection.sort(collection_config["cloud_property"]).first()

    if image is None:
        print("No images found for the specified criteria")
        return

    # Export RGB composite
    rgb_image = image.select(collection_config["rgb_bands"])

    # Set visualization parameters for the RGB export
    vis_params = collection_config["vis_params"]
    rgb_filename = os.path.join(
        output_dir,
        f'{collection_name.replace("/", "_")}_{start_date}_to_{end_date}_rgb.tif',
    )

    # Export RGB composite with visualization parameters
    geemap.ee_export_image(
        rgb_image,
        filename=rgb_filename,
        scale=collection_config["scale"],
        region=geometry,
        file_per_band=False,
    )

    print(f"RGB composite downloaded to: {rgb_filename}")

    # Export individual bands if requested
    if export_separate_bands:
        for band in collection_config["all_bands"]:
            band_image = image.select(band)
            band_filename = os.path.join(
                output_dir,
                f'{collection_name.replace("/", "_")}_{start_date}_to_{end_date}_{band}.tif',
            )

            geemap.ee_export_image(
                band_image,
                filename=band_filename,
                scale=collection_config["scale"],
                region=geometry,
                file_per_band=True,
            )
            print(f"Band {band} downloaded to: {band_filename}")


# Example usage
if __name__ == "__main__":
    # Example bounding box for San Francisco
    bbox = [-122.51, 37.71, -122.35, 37.83]

    # Set output directory
    output_dir = "satellite_images"

    # Download Sentinel-2 imagery
    download_ee_imagery(
        bbox=bbox,
        output_dir=output_dir,
        collection_name="COPERNICUS/S2_SR",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cloud_cover_threshold=20,
        export_separate_bands=True,
    )

    # Download Landsat 9 imagery
    download_ee_imagery(
        bbox=bbox,
        output_dir=output_dir,
        collection_name="LANDSAT/LC09/C02/T1_L2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        cloud_cover_threshold=20,
        export_separate_bands=True,
    )
