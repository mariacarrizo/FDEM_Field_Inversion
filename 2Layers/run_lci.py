"""Laterally Constrained Inversion (LCI) for 2-layer FDEM data.

Replaces the numbered 4_*.py scripts.  Run with --help to see all options.

Examples
--------
Reproduce paper results (sorted data, lam=0):
    python run_lci.py 1 1 c

Nosort variant:
    python run_lci.py 1 1 c --no-sort

With regularisation (lam=1, relaxed conductivity constraint):
    python run_lci.py 1 1 c --lam 1 --cw2 2
"""
import argparse
import numpy as np
import pygimli as pg
import sys
sys.path.insert(1, '../src')

from FDEM1D import FDEM1DModelling, FDEM1DModelling_nosort, LCModelling


def parse_args():
    p = argparse.ArgumentParser(
        description='Run Laterally Constrained Inversion (LCI) on 2-layer FDEM data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('slope',    help='Model slope identifier (e.g. 1)')
    p.add_argument('contrast', help='Conductivity contrast identifier (e.g. 1)')
    p.add_argument('unit',     help='Unit type: c = conductivity [S/m], v = resistivity [Ohm·m]')
    p.add_argument('--no-sort', action='store_true',
                   help='Use unsorted (fixed-array) data instead of mid-point sorted data')
    p.add_argument('--lam', type=float, default=0,
                   help='Inversion regularisation parameter lambda (default: 0)')
    p.add_argument('--cw1', type=float, default=1,
                   help='Lateral constraint weight for layer thicknesses (default: 1)')
    p.add_argument('--cw2', type=float, default=1,
                   help='Lateral constraint weight for conductivities (default: 1)')
    p.add_argument('--thk-min', type=float, default=0.1,
                   help='Lower bound for thickness transform [m] (default: 0.1)')
    p.add_argument('--thk-max', type=float, default=5.0,
                   help='Upper bound for thickness transform [m] (default: 5)')
    p.add_argument('--sig-min', type=float, default=10e-3,
                   help='Lower bound for conductivity transform [S/m] (default: 0.01)')
    p.add_argument('--sig-max', type=float, default=2000e-3,
                   help='Upper bound for conductivity transform [S/m] (default: 2)')
    return p.parse_args()


def main():
    args = parse_args()
    sl, co, cr = args.slope, args.contrast, args.unit
    suffix = '_nosort' if args.no_sort else '_sort'

    print(f'slope={sl}  contrast={co}  unit={cr}  sort={not args.no_sort}  '
          f'lam={args.lam}  cw1={args.cw1}  cw2={args.cw2}')

    model_file = f'models/model_s{sl}_c{co}_u{cr}.npy'
    data_file  = f'data/data_s{sl}_c{co}_u{cr}{suffix}.npy'
    print(f'Loading  {model_file}')
    print(f'         {data_file}')

    model_true = np.load(model_file)
    data_true  = np.load(data_file)
    nLayers = model_true.shape[1] - 1

    model_ini = np.ones_like(model_true)
    model_ini[:, 0] *= 3    # initial thickness
    model_ini[:, 1] *= 0.5  # initial sigma_1
    model_ini[:, 2] *= 0.5  # initial sigma_2

    FopClass = FDEM1DModelling_nosort if args.no_sort else FDEM1DModelling
    LCI = LCModelling(FopClass)

    LCI.initJacobian(dataVals=data_true, nLay=nLayers)
    LCI.createJacobian(model_ini)
    LCI.constraint_matrix(data_true, nLay=nLayers)
    LCI.normalization(data_true, cWeight_1=args.cw1, cWeight_2=args.cw2, nLay=nLayers)
    LCI.createWeight(data_true, cWeight_1=args.cw1, cWeight_2=args.cw2, nLay=nLayers)
    LCI.createConstraints()

    transThk = pg.trans.TransLogLU(args.thk_min, args.thk_max)
    transSig = pg.trans.TransLogLU(args.sig_min, args.sig_max)
    LCI.region(1).setTransModel(transThk)
    LCI.region(2).setTransModel(transSig)

    data_vec = data_true.ravel()
    rel_err  = np.ones_like(data_vec) * 1e-3

    inv = pg.Inversion(LCI)
    model_inv = inv.run(data_vec, rel_err, startModel=model_ini.ravel(),
                        verbose=False, lam=args.lam)

    model_inv = np.array(model_inv).reshape(model_true.shape)
    data_inv  = np.array(inv.response).reshape(data_true.shape)

    tag = f's{sl}_c{co}_u{cr}'
    if args.no_sort:
        tag += '_nosort'
    if args.lam != 0:
        tag += f'_lam{args.lam:g}'

    np.save(f'results/model_inv_{tag}', model_inv)
    np.save(f'results/data_inv_{tag}',  data_inv)
    print(f'\nSaved  results/model_inv_{tag}.npy')
    print(f'       results/data_inv_{tag}.npy')


if __name__ == '__main__':
    main()
