"""
Turbine Specification Loader
==============================

Loads turbine specifications from a CSV database.

Functions:
    - load_turbine: Load a single turbine spec by name → dict

Example:
    from tidal_portfolio.site_processing.turbine import load_turbine

    turbine = load_turbine("RM1")
    print(turbine['rated_power_mw'])  # 1.1
"""

import csv
from pathlib import Path

from tidal_portfolio.config import TURBINE_CSV_PATH


class TurbineNotFoundError(Exception):
    """Raised when a turbine is not found in the CSV database."""
    pass


def load_turbine(turbine_name, csv_path=None):
    """
    Load turbine specification from a CSV database.

    Parameters
    ----------
    turbine_name : str
        Name of the turbine to load (must match DEVICE column in CSV).
    csv_path : str or Path, optional
        Path to the CSV file. Defaults to data/turbine_specifications.csv.

    Returns
    -------
    dict
        Turbine specification with keys: name, rated_power_mw,
        cut_in_speed_ms, rated_speed_ms, cut_out_speed_ms, rotor_diameter_m.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    TurbineNotFoundError
        If the turbine name is not found in the CSV.
    ValueError
        If required columns are missing or values are invalid.
    """
    if csv_path is None:
        csv_path = TURBINE_CSV_PATH
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Turbine database not found: {csv_path}\n"
            f"Please provide a valid path to your turbine specifications CSV file."
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        required_cols = {
            "DEVICE",
            "CUT_IN_SPEED",
            "RATED_SPEED",
            "CUT_OUT_SPEED",
            "RATED_POWER",
            "ROTOR_DIAMETER",
        }
        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {csv_path}")

        available_cols = set(reader.fieldnames)
        missing_cols = required_cols - available_cols
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}\n"
                f"Available columns: {available_cols}"
            )

        # Search for the turbine, collecting names for error message
        available = []
        for row in reader:
            device = row["DEVICE"].strip()
            available.append(device)
            if device == turbine_name:
                return _parse_csv_row(row, turbine_name)

    raise TurbineNotFoundError(
        f"Turbine '{turbine_name}' not found in database.\n"
        f"Available turbines: {available}\n"
        f"Database path: {csv_path}"
    )


def _parse_csv_row(row, turbine_name):
    """Parse a CSV row into a turbine specification dict.

    All fields are required — raises ValueError if any value is
    missing or cannot be parsed as a float.

    Args:
        row: Dict from csv.DictReader with DEVICE, RATED_POWER,
            CUT_IN_SPEED, RATED_SPEED, CUT_OUT_SPEED, ROTOR_DIAMETER.
        turbine_name: Turbine name (for error messages).

    Returns:
        dict with keys: name, rated_power_mw, cut_in_speed_ms,
        rated_speed_ms, cut_out_speed_ms, rotor_diameter_m.

    Raises:
        ValueError: If any required field is missing or non-numeric.
    """

    def parse_float(value, field_name):
        value = value.strip() if value else ""
        if value in ("", "-"):
            raise ValueError(
                f"Missing required value for '{field_name}' in turbine '{turbine_name}'"
            )
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Invalid value '{value}' for '{field_name}' in turbine '{turbine_name}'"
            )

    rated_power_w = parse_float(row["RATED_POWER"], "RATED_POWER")
    rated_power_mw = rated_power_w / 1_000_000

    return {
        "name": turbine_name,
        "rated_power_mw": rated_power_mw,
        "cut_in_speed_ms": parse_float(row["CUT_IN_SPEED"], "CUT_IN_SPEED"),
        "rated_speed_ms": parse_float(row["RATED_SPEED"], "RATED_SPEED"),
        "cut_out_speed_ms": parse_float(row["CUT_OUT_SPEED"], "CUT_OUT_SPEED"),
        "rotor_diameter_m": parse_float(row["ROTOR_DIAMETER"], "ROTOR_DIAMETER"),
    }
