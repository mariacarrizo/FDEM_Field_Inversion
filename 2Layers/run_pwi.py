"""Piecewise (1D sounding-by-sounding) Inversion (PWI) for 2-layer FDEM data.

Replaces the numbered 6_*.py scripts.  Run with --help to see all options.

Examples
--------
Reproduce paper results (sorted data, lam=0):
    python run_pwi.py 1 1 c

Nosort variant:
    python run_pwi.py 1 1 c --no-sort

With regularisation (lam=1):
    python run_pwi.py 1 1 c --lam 1
"""
import argparse
import numpy as np
import pygimli as pg
import sys
sys.path.insert(1, '../src')

from FDEM1D import FDEM1DModelling, FDEM1DModelling_nosort


def parse_args():
    p = argparse.ArgumentParser(
        description='Run Piecewise (1D) Inversion (PWI) on 2-layer FDEM data.',
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

    print(f'slope={sl}  contrast={co}  unit={cr}  sort={not args.no_sort}  lam={args.lam}')

    data_file = f'data/data_s{sl}_c{co}_u{cr}{suffix}.npy'
    print(f'Loading  {data_file}')
    data_true = np.load(data_file)
    npos = len(data_true)

    FopClass = FDEM1DModelling_nosort if args.no_sort else FDEM1DModelling
    fop = FopClass(nlay=2)
    fop.region(0).setTransModel(pg.trans.TransLogLU(args.thk_min, args.thk_max))
    fop.region(1).setTransModel(pg.trans.TransLogLU(args.sig_min, args.sig_max))

    start_model = np.array([3, 0.5, 0.5])
    model_inv = []
    response   = []

    for pos in range(npos):
        dat     = data_true[pos]
        rel_err = np.ones_like(dat) * 1e-3
        inv = pg.Inversion(fop)
        model_inv.append(inv.run(dataVals=dat, errorVals=rel_err,
                                 startModel=start_model, verbose=False, lam=args.lam))
        response.append(inv.response)

    tag = f's{sl}_c{co}_u{cr}'
    if args.no_sort:
        tag += '_nosort'
    if args.lam != 0:
        tag += f'_lam{args.lam:g}'

    np.save(f'results/model_inv_{tag}_1D', model_inv)
    np.save(f'results/data_inv_{tag}_1D',  response)
    print(f'\nSaved  results/model_inv_{tag}_1D.npy')
    print(f'       results/data_inv_{tag}_1D.npy')


if __name__ == '__main__':
    main()
