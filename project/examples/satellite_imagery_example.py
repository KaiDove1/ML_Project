# Example usage
from project.satellite_imagery.download import download_bands

if __name__ == "__main__":
    # Example bounding box for San Francisco
    bbox = [-122.51, 37.71, -122.35, 37.83]

    # Set output directory
    output_dir = "data/satellite_images/california/san_francisco"

    # Download Landsat 9 imagery
    download_bands(
        bbox=bbox,
        output_dir=output_dir,
        collection_name="LANDSAT/LC09/C02/T1_L2",
        band_names=["B4", "B3", "B2"],
        scale=30,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
