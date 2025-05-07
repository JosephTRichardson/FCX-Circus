"""Utility functions for timestamps."""

import numpy as np
import xarray as xr
import re
import os
import random
import datetime
from typing import Optional

def has_wraparound(time_array: np.ndarray) -> bool:
    """Returns True if the array wraps around (i.e., drops backward).
    Some `time` arrays may be suspect, i.e. they wrap around to 0 at 24 hours."""
    return np.any(np.diff(time_array) < 0)

def resolve_time_hours(hr: np.ndarray, base_date: np.datetime64) -> np.ndarray:
    """
    Given an array of hours-since-midnight and a base date, return absolute UTC timestamps.
    Corrects for midnight wraparound if needed.

    Parameters:
    - hr: 1D array of float hours since 00:00 UTC
    - base_date: np.datetime64[D], e.g., '2017-05-17'

    Returns:
    - 1D array of np.datetime64[s] timestamps
    """
    hr = np.array(hr, dtype=np.float64)

    # Detect wraparound
    diff = np.diff(hr)
    wrap_points = np.where(diff < 0)[0]

    if wrap_points.size > 0:
        # Each wrap indicates a new day, i.e. hour rollover
        day_shifts = np.zeros_like(hr, dtype=int)
        for i, wrap_index in enumerate(wrap_points, start=1):
            day_shifts[wrap_index + 1:] += 1
        hr += day_shifts * 24

    # Convert to datetime64
    delta = (hr * 3600).astype("timedelta64[s]")
    timestamps = base_date.astype("datetime64[s]") + delta

    return timestamps

def get_date_hint(attrs: dict, filename: str) -> Optional[np.datetime64]:
    """
    Attempt to get a date from NetCDF attributes or fall back to extracting from filename.

    Returns:
    - np.datetime64 or None if no valid date can be found.
    """
    # Try 'date' in attributes (e.g., "20170517")
    if "date" in attrs:
        date_str = re.sub(r"[^0-9]", "", str(attrs["date"]))
        if len(date_str) == 8:
            try:
                return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
            except ValueError:
                pass
        elif m := re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            year, month, day = map(int, m.groups())
            try:
                return np.datetime64(f"{year}-{month}-{day}")
            except ValueError:
                pass
        # May be other common formats we need to support

    # Fallback: extract YYYYMMDD from filename
    if m := re.search(r"((?:20|19)\d{6})", filename):
        date_str = m.group(1)
        try:
            return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
        except ValueError:
            pass
    elif m := re.search(r"^((?:20|19)\d{2}-\d{2}-\d{2})$", filename):
        date_str = m.group(1)
        try:
            return np.datetime64(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
        except ValueError:
            pass

    return None  # Nothing usable

def detect_time_var_type(time_var: xr.DataArray) -> dict:
    """
    Analyze the time variable from a dataset and infer its format and reference point.

    Returns a dict with:
        - 'type': one of ['datetime64', 'hours-since-midnight', 'seconds-since-midnight', 'unknown']
        - 'base_time': np.datetime64 or None (if it can be inferred)
        - 'time_array': np.ndarray of np.datetime64[s] if conversion succeeded, else original
        - 'wrapped': True if time resets to 0 after 24h (or 86400s), False otherwise
    """
    result = {
        "type": "unknown",
        "base_time": None,
        "time_array": time_var.values,
        "wrapped": False
    }

    values = time_var.values

    # Case 1: already datetime64
    if np.issubdtype(values.dtype, np.datetime64):
        result["type"] = "datetime64"
        result["time_array"] = values
        return result

    # Case 2: check for units metadata
    units = time_var.attrs.get("units", "").lower()

    # Attempt to guess base time
    match = None
    if "since" in units:
        import re
        match = re.search(r"(seconds|hours)\s+since\s+([0-9T:\- ]+)", units)
        if match:
            unit_str, base_str = match.groups()
            try:
                base_time = np.datetime64(base_str.strip())
                result["base_time"] = base_time
                if "hour" in unit_str:
                    delta = (values * 3600).astype("timedelta64[s]")
                elif "second" in unit_str:
                    delta = values.astype("timedelta64[s]")
                else:
                    delta = None
                if delta is not None:
                    result["time_array"] = base_time + delta
                    result["type"] = f"{unit_str}-since"
                    return result
            except Exception:
                # Don't want to break
                pass

    # Case 3: try to detect wrapped floating time (like 2.5, 3.1, 0.1...)
    if values.dtype.kind in {"f", "i"} and values.ndim == 1:
        if has_wraparound(values):
            result["wrapped"] = True

        if np.max(values) < 25:  # probably hours since midnight
            result["type"] = "hours-since-midnight"
        elif np.max(values) < 86400:  # maybe seconds since midnight
            result["type"] = "seconds-since-midnight"
        else:
            result["type"] = "numeric-time"

    return result

def normalize_timestamps(time_array: xr.DataArray, date_hint: Optional[np.datetime64]) -> xr.DataArray:
    """
    Normalize a time array to datetime64[s], preserving structure and marking it as normalized.
    """
    info = detect_time_var_type(time_array)
    values = time_array.values

    def wrap(result):
        out = xr.DataArray(result, dims=time_array.dims, coords=time_array.coords)
        out.attrs = dict(time_array.attrs)  # preserve existing metadata
        out.attrs["normalized"] = True
        return out

    if np.issubdtype(values.dtype, np.datetime64):
        return wrap(values.astype("datetime64[s]"))

    if info["type"] in {"hours-since", "seconds-since"} and info["base_time"] is not None:
        base = info["base_time"].astype("datetime64[s]")
        delta = (values * 3600).astype("timedelta64[s]") if "hour" in info["type"] else values.astype("timedelta64[s]")
        return wrap(base + delta)

    if info["type"] in {"hours-since-midnight", "seconds-since-midnight"}:
        if date_hint is None:
            raise ValueError("Missing date_hint for relative time array")
        base = date_hint.astype("datetime64[s]")
        delta = (values * 3600).astype("timedelta64[s]") if "hour" in info["type"] else values.astype("timedelta64[s]")
        return wrap(base + delta)

    raise ValueError(f"Cannot normalize time array: unrecognized or unsupported time format ({info['type']}).")

def generate_timestamped_filename(prefix: str, extension: str = "") -> str:
    """Generate a timestamped and UID-tagged filename or folder name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = f"{random.randint(0, 99999999):08d}"
    base_name = f"{prefix}_{timestamp}_{uid}"
    if extension:
        filename = f"{base_name}.{extension.lstrip('.')}"
    else:
        filename = base_name
    return filename
