import json
import re
from collections import defaultdict
from typing import Optional, cast

import pandas as pd

from project.data.dirs import DATA_PATH, PATCHES_DIR

FIELD_CROPS_RAW_DATAFRAME = pd.read_csv(
    DATA_PATH / "census_data" / "table25_parsed.csv"
)
FRUITS_RAW_DATAFRAME = pd.read_csv(DATA_PATH / "census_data" / "table31_parsed.csv")


def get_county_slug(county_name: str):
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
        if not (PATCHES_DIR / get_county_slug(county_name)).exists():
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


def generate_crop_ontology():
    # Generate an ontology.
    crop_ontology = []
    id_counter = 0

    # Identify the units of measurement for each crop.
    for crop in sorted(FIELD_CROPS_RAW_DATAFRAME["crop"].unique()):
        original_name = crop
        crop, units_of_measurement = parse_field_crop_name(crop)
        crop_slug = re.sub(r"[^\w]+", "_", crop).lower().strip()
        crop_ontology.append(
            {"original_name": original_name, "slug": crop_slug, "id": id_counter}
        )
        id_counter += 1

    for crop in sorted(FRUITS_RAW_DATAFRAME["crop"].unique()):
        original_name = crop
        crop = crop.replace(" (See Text)", "")
        crop_slug = re.sub(r"[^\w]+", "_", crop).lower().strip()
        crop_ontology.append(
            {"original_name": original_name, "slug": crop_slug, "id": id_counter}
        )
        id_counter += 1

    return crop_ontology


def generate_clean_dataframe():
    # 1. Load crop class names.
    with open(DATA_PATH / "crop_ontology.json", "r") as f:
        crop_ontology = json.load(f)

    name_order = [ont["slug"] for ont in crop_ontology]
    name_to_slug = {ont["original_name"]: ont["slug"] for ont in crop_ontology}

    # 2. Create a dataframe where row = county_slug and col = crop yield.
    rows_by_county = defaultdict(
        lambda: {ont["slug"]: cast(Optional[float], None) for ont in crop_ontology}
    )

    for county_name, county_df in FIELD_CROPS_RAW_DATAFRAME.groupby("county"):
        county_name = cast(str, county_name)
        county_slug = get_county_slug(county_name)
        for i, entry in county_df.iterrows():
            crop_name: str = entry["crop"]
            crop_slug: str = name_to_slug[crop_name]
            qty: float = entry["harvested_quantity_2022"]
            acres: float = entry["harvested_acres_2022"]

            if acres == 0:
                per_acre_qty = float("nan")
            else:
                per_acre_qty: float = qty / acres

            rows_by_county[county_slug][crop_slug] = per_acre_qty

    rows = [{"county": county, **row} for county, row in rows_by_county.items()]

    df = pd.DataFrame(rows, columns=["county", *name_order])
    df.set_index("county", inplace=True)
    df.to_csv(DATA_PATH / "va_crops_2022.csv")


if __name__ == "__main__":
    generate_clean_dataframe()
