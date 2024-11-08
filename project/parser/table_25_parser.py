import re
import csv

input_file = "../../data/census_data/table25.txt"
output_file = "parsed_table_25.csv"

crop_type_pattern = re.compile(r"^[A-Z ]+ \(.*\)")
data_line_pattern = re.compile(r"^[A-Za-z].*:\s+\d+|[A-Za-z].*:\s+-\s+")

# Storage for parsed data
parsed_data = []
current_crop_type = None

def parse_line(line):
    """Parse a line of data using fixed positions to match table columns."""
    try:
        # Extract columns based on fixed positions
        area = line[:50].strip()
        farms_2022 = line[50:55].strip()
        acres_2022 = line[56:66].strip()
        quantity_2022 = line[67:78].strip()
        irrigated_farms_2022 = line[79:84].strip()
        irrigated_acres_2022 = line[85:95].strip()
        farms_2017 = line[96:101].strip()
        acres_2017 = line[102:112].strip()
        quantity_2017 = line[113:123].strip()
        irrigated_farms_2017 = line[124:129].strip()
        irrigated_acres_2017 = line[130:140].strip()

        # Replace any missing data (e.g., "-", "(D)") with None
        return {
            "geographic_area": area,
            "2022_farms": farms_2022 if farms_2022 not in ("-", "(D)") else None,
            "2022_acres": acres_2022 if acres_2022 not in ("-", "(D)") else None,
            "2022_quantity": quantity_2022 if quantity_2022 not in ("-", "(D)") else None,
            "2022_irrigated_farms": irrigated_farms_2022 if irrigated_farms_2022 not in ("-", "(D)") else None,
            "2022_irrigated_acres": irrigated_acres_2022 if irrigated_acres_2022 not in ("-", "(D)") else None,
            "2017_farms": farms_2017 if farms_2017 not in ("-", "(D)") else None,
            "2017_acres": acres_2017 if acres_2017 not in ("-", "(D)") else None,
            "2017_quantity": quantity_2017 if quantity_2017 not in ("-", "(D)") else None,
            "2017_irrigated_farms": irrigated_farms_2017 if irrigated_farms_2017 not in ("-", "(D)") else None,
            "2017_irrigated_acres": irrigated_acres_2017 if irrigated_acres_2017 not in ("-", "(D)") else None,
        }
    except IndexError:
        return None  # Return None if line format doesn't match expected structure

# Open the file and parse its content
with open(input_file, "r") as file:
    for line in file:
        line = line.strip()

        # Check if line is a crop type
        if crop_type_pattern.match(line):
            current_crop_type = line.strip()
            continue

        # Check if line contains geographic area data
        if data_line_pattern.match(line):
            parsed_line = parse_line(line)
            if parsed_line:
                parsed_line["crop_type"] = current_crop_type
                parsed_data.append(parsed_line)

# Write the parsed data to a CSV file
with open(output_file, "w", newline="") as csvfile:
    fieldnames = [
        "crop_type", "geographic_area", "2022_farms", "2022_acres", "2022_quantity",
        "2022_irrigated_farms", "2022_irrigated_acres", "2017_farms", "2017_acres",
        "2017_quantity", "2017_irrigated_farms", "2017_irrigated_acres"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(parsed_data)

print(f"Parsed data has been saved to {output_file}")