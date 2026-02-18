#!/usr/bin/env python3
"""
Adjust Cd column in polar CSV files:
- alpha < 0: no change
- 0 <= alpha <= 10: linear ramp from *1.0 at alpha=0 to *1.5 at alpha=10
- alpha > 10: multiply by 1.5
"""

import csv
import os
from pathlib import Path


def calculate_cd_multiplier(alpha):
    """Calculate the Cd multiplier based on alpha value."""
    if alpha < 0:
        return 1.0
    elif alpha <= 10:
        # Linear ramp from 1.0 at alpha=0 to 1.5 at alpha=10
        return 1.0 + (alpha / 10.0) * 0.5
    else:
        return 1.5


def adjust_cd_in_file(filepath):
    """Adjust Cd column in a single CSV file."""
    # Read the file
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            alpha = float(row["alpha"])
            cd_original = float(row["Cd"])
            multiplier = calculate_cd_multiplier(alpha)
            row["Cd"] = cd_original * multiplier
            rows.append(row)

    # Write back to file
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Adjusted {filepath.name}")


def main():
    # Get the directory containing the script
    script_dir = Path(__file__).parent
    polar_dir = script_dir / "2D_polars_CFD_drag_increase"

    # Process all CSV files
    csv_files = sorted(polar_dir.glob("*.csv"))

    for csv_file in csv_files:
        adjust_cd_in_file(csv_file)

    print(f"\nProcessed {len(csv_files)} files.")


if __name__ == "__main__":
    main()
