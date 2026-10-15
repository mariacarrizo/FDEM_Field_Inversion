"""Reader for raw DUALEM-842s .xyz field exports.

The .xyz format has a few '\\'-prefixed metadata/header lines followed by
whitespace-separated data rows, e.g.:

    \\Dualem Filename is Default
    \\Dualem Sensor Length Code is **
    \\..STN.SP.LINE.LP.HC2-mS/m.HI2-ppt.PC2-mS/m.PI2-ppt..HC4-mS/m. ...
      7103 N     0 E  2250.80  -253.19  1264.30  -115.78    -2.20  ...

The column-name header line has one name per data column, joined by '.';
runs of '..' are just fixed-width alignment padding between names and are
dropped. The DUALEM-842s only records H (horizontal coplanar) and P
(perpendicular) coil pairs -- there is no V (vertical coplanar) channel in
this export, unlike the synthetic data used elsewhere in this repo.
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'src'))
from FDEM1D import InstrumentConfig

# DUALEM-842s field configuration: H + P coils only, 2/4/8 m (H) and
# 2.1/4.1/8.1 m (P) spacings, 9 kHz. Passed to FDEM1DModelling_field so the
# forward model's channel layout matches DATA_COLUMNS below.
FIELD_CONFIG = InstrumentConfig(
    frequency=9000.0,
    coil_spacings=[2.0, 4.0, 8.0],
    perp_spacings=[2.1, 4.1, 8.1],
    coil_orientations=['H', 'P'],
    height=0.47,
)

# Data columns fed to the inversion, in the exact order FDEM1D_field(...)
# returns them for coil_orientations=['H', 'P']: H then P quadrature at
# 2/4/8 m, then H in-phase at 4/8 m only (2 m in-phase is dropped, matching
# the convention already used for Field/data/Field_data.npy).
DATA_COLUMNS = ['HC2-mS/m', 'HC4-mS/m', 'HC8-mS/m',
                'PC2-mS/m', 'PC4-mS/m', 'PC8-mS/m',
                'HI4-ppt', 'HI8-ppt']

POSITION_COLUMNS = ['STN', 'LINE', 'Lat', 'Lon', 'GPS_Alt']


def _header_columns(line):
    """Split a '\\'-prefixed header line on '.' into column names."""
    return [tok for tok in line.strip().lstrip('\\').split('.') if tok]


def read_xyz(path):
    """Read a raw DUALEM .xyz file into a DataFrame, one row per sounding.

    Every column is coerced to numeric except 'SP' and 'LP', which hold
    single-letter direction flags (e.g. 'N'/'E') rather than measurements.
    """
    columns = None
    rows = []
    with open(path, 'r') as fh:
        for raw_line in fh:
            line = raw_line.rstrip('\n')
            if not line.strip():
                continue
            if line.lstrip().startswith('\\'):
                tokens = _header_columns(line)
                if columns is None and 'STN' in tokens:
                    columns = tokens
                continue
            rows.append(line.split())

    if columns is None:
        raise ValueError(f"Could not find the 'STN' column header line in {path}")

    bad_rows = [i for i, r in enumerate(rows) if len(r) != len(columns)]
    if bad_rows:
        raise ValueError(
            f"{len(bad_rows)} data row(s) in {path} do not match the "
            f"{len(columns)} columns found in the header (first bad row: {bad_rows[0] + 1})"
        )

    df = pd.DataFrame(rows, columns=columns)
    for col in columns:
        if col in ('SP', 'LP'):
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def build_data_matrix(df):
    """Build the (N, 8) inversion data array from a DataFrame returned by
    read_xyz(). Raw mS/m and ppt values are divided by 1000 to match the
    dimensionless fraction FDEM1D_field expects (see Field_data.npy, whose
    'Q'/'IP' columns are the same DUALEM channels already scaled this way).
    """
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Field file is missing expected columns: {missing}")
    return df[DATA_COLUMNS].to_numpy(dtype=float) / 1000.0


def has_gps_fix(df):
    """Boolean mask: True where Lat/Lon look like a real GPS fix (not 0,0)."""
    return ~((df['Lat'] == 0) & (df['Lon'] == 0))
