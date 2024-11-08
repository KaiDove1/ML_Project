import json

import pandas as pd
import numpy as np


def load_counties_by_state() -> dict[str, list[dict]]:
    df = pd.read_csv("data/geojson/fips_codes.csv")
    code = df["STATE"]
    state_name = df["STATE_NAME"]
    code_to_state = {code: state_name for code, state_name in zip(code, state_name)}

    results: dict[str, list] = {}

    with open("data/geojson/counties.geojson") as f:
        data = json.load(f)
        for feature in data["features"]:
            properties = feature["properties"]
            state_code = int(properties["STATEFP"])
            county_name = properties["NAME"]
            state_name = code_to_state[state_code]
            if state_name not in results:
                results[state_name] = []

            # lonlat = np.array(sum(feature["geometry"]["coordinates"], start=[]))
            if type(feature["geometry"]["coordinates"][0][0][0]) == list:
                lonlat = np.array(
                    sum(sum(feature["geometry"]["coordinates"], start=[]), start=[])
                )
            else:
                lonlat = np.array(sum(feature["geometry"]["coordinates"], start=[]))

            lon = lonlat[:, 0]
            lat = lonlat[:, 1]

            results[state_name].append(
                {
                    "name": county_name,
                    "bbox": [
                        float(x) for x in [lon.min(), lat.min(), lon.max(), lat.max()]
                    ],
                }
            )

    return results
