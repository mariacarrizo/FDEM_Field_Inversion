# FDEM Field Inversion

Standalone toolkit for running Laterally Constrained Inversion (LCI) and
Piecewise Inversion (PWI) on frequency-domain EMI (FDEM) data, extracted
from the [FDEM_LCI_PWI](https://doi.org/10.1016/j.jappgeo.2026.106130)
research repo. Two independent use cases:

- **`Field/`** — run LCI/PWI directly on raw **DUALEM-842s `.xyz`** field
  exports. Ready to run out of the box against `Field/data/sample.xyz`.
- **`2Layers/`** — the original synthetic 2-layer LCI/PWI scripts, kept as
  a reference implementation / CLI pattern. These expect their own
  `model_s*_c*_u*.npy` / `data_s*_c*_u*.npy` files (not included here — see
  [Notes](#2layers-notes) below).

Both use cases share the forward-modelling code in `src/FDEM1D.py`.

## Installation

```bash
conda env create -f environment.yml
conda activate fdem_lci_pwi
```

or, without conda:

```bash
pip install -r requirements.txt
```

## Structure

```
.
├── environment.yml / requirements.txt
├── src/
│   ├── FDEM1D.py          forward model + pygimli Modelling/LCModelling classes
│   └── showStitched.py    stitched 1D-model cross-section plotting
├── Field/
│   ├── README.md          full usage guide (start here for field data)
│   ├── dualem_xyz.py      DUALEM-842s .xyz reader
│   ├── run_lci_field.py   LCI on raw .xyz field data
│   ├── run_pwi_field.py   PWI on raw .xyz field data
│   └── data/sample.xyz    small example file for smoke-testing
└── 2Layers/
    ├── run_lci.py         LCI on synthetic 2-layer data
    └── run_pwi.py         PWI on synthetic 2-layer data
```

## Quick start (field data)

```bash
cd Field
python run_lci_field.py data/sample.xyz
python run_pwi_field.py data/sample.xyz
```

See [`Field/README.md`](Field/README.md) for the full guide: input format,
all CLI options, output files, and how to use `dualem_xyz.py` directly.

## Selecting coil configurations and offsets

The forward models in `src/FDEM1D.py` can use any subset of the instrument's
coils and offsets. Available coils and offsets (DUALEM-842s):

| Coil | Offsets [m]       |
|------|-------------------|
| `H`  | 2.0, 4.0, 8.0     |
| `V`  | 2.0, 4.0, 8.0     |
| `P`  | 2.1, 4.1, 8.1     |

Use `InstrumentConfig.subset(coils=..., offsets=...)`. Offsets are matched on
their integer part, so `2` selects H/V at 2.0 m **and** P at 2.1 m.

```python
from FDEM1D import FDEM1D, DUALEM842S

cfg = DUALEM842S.subset(coils=['H'], offsets=[2, 4])        # H at 2.0 and 4.0 m
cfg = DUALEM842S.subset(coils=['H', 'P'], offsets=[8])      # H 8.0 m, P 8.1 m
cfg = DUALEM842S.subset(coils=['V'])                        # V, all offsets
cfg = DUALEM842S.subset(offsets=[4, 8])                     # all coils, 4 and 8 m

resp = FDEM1D(sgm, thk, config=cfg)
```

For different offsets per coil, give them explicitly:

```python
from FDEM1D import InstrumentConfig
cfg = InstrumentConfig(offsets={'H': [2.0, 4.0], 'V': [8.0], 'P': [4.1]})
```

The config works with `FDEM1D`, `FDEM1D_mdp`, `FDEM1D_field`, `FDEM1D_topo`
and the `FDEM1DModelling*` classes (`config=cfg`).

Notes:

- The response vector keeps the order `[HOP, POP, VOP, HIP, PIP, VIP]`
  (`FDEM1D_field`: `[HOP, VOP, POP, HIP, VIP]`), containing only the selected
  channels. Select the observed data the same way, or the sizes won't match.
- `FDEM1D_field` never includes 2 m in-phase (not recorded in field data), so
  H or V at 2 m only gives quadrature values.
- Asking for a coil with no matching offset raises a `ValueError`.

## `2Layers` notes

`run_lci.py` and `run_pwi.py` take a `slope`/`contrast`/`unit` triple and
load `models/model_s{slope}_c{contrast}_u{unit}.npy` and
`data/data_s{slope}_c{contrast}_u{unit}[_nosort].npy` relative to
`2Layers/`. Those synthetic dataset files aren't included in this toolkit —
supply your own in that shape (see the docstrings/`--help` in each script
for the exact CLI), or generate them with the model/forward-modelling
scripts in the original
[FDEM_LCI_PWI](https://doi.org/10.1016/j.jappgeo.2026.106130) repo.

```bash
cd 2Layers
python run_lci.py 1 1 c --help
python run_pwi.py 1 1 c --help
```
