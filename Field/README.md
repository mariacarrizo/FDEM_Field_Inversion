# Field

Run LCI and PWI inversions directly on raw DUALEM-842s `.xyz` field exports
(no pre-processing into `.npy` required). This replaces the manual "load
`Field_data.npy`" step in `15_LCI_Field.ipynb` / `15_Piecewise_Field.ipynb`
with a reader for the instrument's native output format.

| File | Purpose |
|---|---|
| `dualem_xyz.py` | Shared `.xyz` reader + DUALEM-842s field config. Imported by both scripts below, not run directly. |
| `run_lci_field.py` | Laterally Constrained Inversion (LCI) — inverts each survey line jointly. |
| `run_pwi_field.py` | Piecewise Inversion (PWI) — inverts each sounding independently. |
| `data/sample.xyz` | Small example file (30 soundings) for smoke-testing. |

## Input format

A raw DUALEM-842s `.xyz` export: a few `\`-prefixed metadata lines, one
`\`-prefixed column-header line (names joined by `.`, e.g.
`\..STN.SP.LINE.LP.HC2-mS/m.HI2-ppt...`), then one whitespace-separated data
row per sounding. See `data/sample.xyz` for a real example.

The DUALEM-842s only records **H (horizontal coplanar)** and **P
(perpendicular)** coil pairs at 2/4/8 m spacings — there is no V (vertical
coplanar) channel in this export. `dualem_xyz.FIELD_CONFIG` reflects that,
and the inversion data vector is built from:

```
HC2, HC4, HC8, PC2, PC4, PC8   (quadrature / apparent conductivity, mS/m)
HI4, HI8                      (in-phase, ppt — 2 m in-phase is dropped,
                                matching the convention already used for
                                Field/data/Field_data.npy)
```

Raw mS/m and ppt values are divided by 1000 before inversion to match the
dimensionless fraction the forward model (`FDEM1D_field` in `src/FDEM1D.py`)
returns.

## `dualem_xyz.py`

Not a script — a library used by both `run_*_field.py` scripts. Useful
directly if you want the parsed data for something else (e.g. a notebook):

```python
import sys
sys.path.insert(1, '.')  # or wherever Field/ is on your machine
from dualem_xyz import read_xyz, build_data_matrix, has_gps_fix

df = read_xyz('data/sample.xyz')       # one row per sounding, all columns
data = build_data_matrix(df)           # (N, 8) array ready for FDEM1D_field
fix_ok = has_gps_fix(df)                # True where Lat/Lon isn't (0, 0)
```

`read_xyz` raises with a clear message if it can't find the `STN`
column-header line, or if a data row's token count doesn't match the header
(a sign the file has been hand-edited or truncated).

## `run_lci_field.py`

Groups soundings by their `LINE` number and inverts each line jointly with
lateral constraints between neighbouring soundings.

```bash
python run_lci_field.py data/sample.xyz
python run_lci_field.py survey.xyz --nlay 2 --lam 1 --cw2 2
python run_lci_field.py survey.xyz -o results/ackerdijkse --plot
python run_lci_field.py --help
```

Key options:

| Flag | Default | Meaning |
|---|---|---|
| `xyz_file` | — | Raw DUALEM `.xyz` field file (positional) |
| `-o, --out-prefix` | `results/<xyz filename>_lci` | Output file prefix |
| `--nlay` | `2` | Number of layers |
| `--height` | `0.47` | Instrument height above ground [m] |
| `--lam` | `1.0` | Regularisation parameter (matches the field notebook) |
| `--cw1` / `--cw2` | `1.0` / `1.0` | Lateral constraint weight: thickness / conductivity |
| `--thk-min/--thk-max` | `0.1` / `5.0` | Thickness transform bounds [m] |
| `--sig-min/--sig-max` | `1e-3` / `500e-3` | Conductivity transform bounds [S/m] |
| `--rel-error` | `1e-3` | Assumed relative data error |
| `--plot` | off | Save a stitched cross-section PNG per line to `<out-prefix>_figures/` |

A line with fewer than 2 soundings is skipped (LCI needs at least one
neighbour pair) and reported to stdout; its rows stay `NaN` in the output.

**Output** (written to `<out-prefix>`):
- `.csv` — one row per sounding: STN / LINE / Lat / Lon / GPS_Alt (whichever
  are present in the input) plus `h_1...h_{nlay-1}` [m] and
  `sigma_1_mSm...sigma_{nlay}_mSm` [mS/m]
- `_model.npy` — raw inverted model array, `(n_soundings, 2*nlay-1)`,
  thicknesses in m then conductivities in **S/m**
- `_response.npy` — forward response of the inverted model, same shape/units
  as the input data vector (for QC against the raw data)

## `run_pwi_field.py`

Inverts every sounding independently — no lateral constraint, so `LINE`
grouping is only used for the printed per-line RMSE, not for the inversion
itself.

```bash
python run_pwi_field.py data/sample.xyz
python run_pwi_field.py survey.xyz --nlay 2 --lam 1
python run_pwi_field.py survey.xyz -o results/ackerdijkse
python run_pwi_field.py --help
```

Same flags as `run_lci_field.py` minus `--cw1`/`--cw2`/`--plot` (there's no
lateral constraint or line-based plot to configure), and `--lam` defaults to
`0.0` to match `15_Piecewise_Field.ipynb`. Same output convention
(`.csv`, `_model.npy`, `_response.npy`) with an `_pwi` suffix.

## Trying it out

```bash
cd Field
python run_lci_field.py data/sample.xyz
python run_pwi_field.py data/sample.xyz
```

Both need the project's `src/` on the path, which the scripts already
handle via `sys.path.insert`, so just run them from inside `Field/`.

## Environment note (Windows)

`pygimli` has no Windows build on conda-forge — only PyPI wheels. Install it
via the `pip:` section of `environment.yml` (already set up that way), not
as a conda dependency, and keep `numpy<2` (the compiled `pgcore` extension
pygimli depends on is built against the NumPy 1.x ABI).

Even with that fixed, `pg.Inversion(...).run()` has been observed to
segfault natively on Windows with the pip wheels (tested with pygimli 1.5.0
and 1.6.0) — this reproduces with the repo's own pre-existing
`2Layers/run_lci.py`/`run_pwi.py` too, so it isn't specific to the scripts
in this folder. If you hit a silent crash with no Python traceback, that's
almost certainly it — pygimli's compiled core is developed and tested
primarily on Linux/Mac via conda-forge, so WSL or a Linux/Mac machine is the
most reliable way to actually run these inversions today.
