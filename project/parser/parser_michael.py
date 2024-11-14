import re
from enum import Enum

import pandas as pd

TABLE_25_COLUMN_NAMES = [
    "harvested_farms_2022",
    "harvested_acres_2022",
    "harvested_quantity_2022",
    "irrigated_farms_2022",
    "irrigated_acres_2022",
    "harvested_farms_2017",
    "harvested_acres_2017",
    "harvested_quantity_2017",
    "irrigated_farms_2017",
    "irrigated_acres_2017",
]


TABLE_31_COLUMN_NAMES = [
    "total_farms_2022",
    "total_acres_2022",
    "total_bearing_age_farms_2022",
    "total_bearing_age_acres_2022",
    "total_nonbearing_age_farms_2022",
    "total_nonbearing_age_acres_2022",
    "total_farms_2017",
    "total_acres_2017",
    "total_bearing_age_farms_2017",
    "total_bearing_age_acres_2017",
    "total_nonbearing_age_farms_2017",
    "total_nonbearing_age_acres_2017",
]


class State(Enum):
    INITIAL = 0
    PARSING_CROP_NAME = 1
    PARSING_STATE_TOTAL = 2
    PARSING_COUNTIES = 3
    CONTINUED_TABLE_HEADER = 4


"""
Different types of row values:
 - (D): Omitted to prevent disclosure of values for individual farms.
 - "-": Zero.
 - (NA): Not available.
"""


def standardize_cell_value(cell_value: str):
    if re.match(r"\d{1,3}(,\d{3})*", cell_value):
        return int(cell_value.replace(",", ""))

    if cell_value in ["(D)", "(NA)"]:
        return "NA"

    if cell_value == "-":
        return 0

    if cell_value == "(Z)":
        return 0

    raise ValueError(f"Unexpected cell value: {cell_value}")


def parse_va_census_table(lines: list[str], column_names: list[str]) -> pd.DataFrame:
    state = State.INITIAL
    crop = ""
    data = []
    for lineno, line in enumerate(lines):
        if line.count(":") != 1 or re.match(r"Table \d+\.+.+", line):
            assert state in [
                State.INITIAL,
                State.PARSING_COUNTIES,
                State.CONTINUED_TABLE_HEADER,
            ], f"Unexpected line at lineno {lineno}: {line}"

            match state:
                case State.INITIAL | State.CONTINUED_TABLE_HEADER:
                    pass

                case State.PARSING_COUNTIES:
                    state = State.CONTINUED_TABLE_HEADER

            continue

        geographic_area_part, data_part = line.split(":")
        geographic_area_part = geographic_area_part.strip()
        data_part = data_part.strip()

        match state:
            case State.INITIAL | State.PARSING_CROP_NAME:
                if geographic_area_part == "":
                    continue

                is_uppercase = geographic_area_part.upper() == geographic_area_part
                if is_uppercase:
                    if len(crop) > 0 and not crop.endswith(" "):
                        crop += " "
                    crop += geographic_area_part.title()
                    continue

                if geographic_area_part == "State Total":
                    state = State.PARSING_STATE_TOTAL
                    continue

                raise ValueError(f"Unexpected line at lineno {lineno}: {line}")

            case State.PARSING_STATE_TOTAL:
                if (
                    geographic_area_part == "Virginia"
                    or re.match(r"Virginia \.+", geographic_area_part)
                    or geographic_area_part == ""
                ):
                    continue

                if geographic_area_part == "Counties":
                    state = State.PARSING_COUNTIES
                    continue

                raise ValueError(f"Unexpected line at lineno {lineno}: {line}")

            case State.PARSING_COUNTIES:
                if geographic_area_part == "":
                    continue

                if "Con." in geographic_area_part:
                    continue

                is_uppercase = (
                    geographic_area_part.upper() == geographic_area_part
                ) and geographic_area_part != ""
                if is_uppercase:
                    crop = geographic_area_part.title()
                    state = State.PARSING_CROP_NAME
                    continue

                # Parse the crop data. This is the majority of the lines.
                county = geographic_area_part.rstrip(".").rstrip(" ")
                row_data: list[str] = re.split(r"\s+", data_part)

                assert len(row_data) == len(
                    column_names
                ), f"Unexpected number of columns. Got: {row_data}"

                data.append(
                    {
                        "crop": crop,
                        "county": county,
                        **(
                            {
                                column_names[i]: standardize_cell_value(row_data[i])
                                for i in range(len(column_names))
                            }
                        ),
                    }
                )

            case State.CONTINUED_TABLE_HEADER:
                if geographic_area_part == "":
                    continue

                if geographic_area_part == "Counties - Con.":
                    state = State.PARSING_COUNTIES
                    continue

                elif "Con." in geographic_area_part:
                    continue

                elif geographic_area_part.upper() == geographic_area_part:
                    continue

                raise ValueError(f"Unexpected line at lineno {lineno}: {line}")

    return pd.DataFrame(data)


def main():
    # We will go line-by-line through the file.
    with open("data/census_data/table25_clean.txt") as f:
        lines = f.read().split("\n")

    df = parse_va_census_table(lines, TABLE_25_COLUMN_NAMES)
    df.to_csv("data/census_data/table25_parsed.csv", index=False)

    # We will go line-by-line through the file.
    with open("data/census_data/table31_clean.txt") as f:
        lines = f.read().split("\n")

    df = parse_va_census_table(lines, TABLE_31_COLUMN_NAMES)
    df.to_csv("data/census_data/table31_parsed.csv", index=False)


if __name__ == "__main__":
    main()
