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
