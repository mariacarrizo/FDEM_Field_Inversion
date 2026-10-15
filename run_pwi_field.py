"""Piecewise (sounding-by-sounding) Inversion (PWI) on raw DUALEM-842s .xyz
field data.

Inverts each sounding independently (no lateral constraint between
neighbours), exactly like 15_Piecewise_Field.ipynb but reading straight
from a raw .xyz export instead of the pre-built Field_data.npy.

Examples
--------
    python run_pwi_field.py survey.xyz
    python run_pwi_field.py survey.xyz --nlay 2 --lam 1
    python run_pwi_field.py survey.xyz -o results/ackerdijkse
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pygimli as pg

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'src'))
from FDEM1D import FDEM1DModelling_field

from dualem_xyz import read_xyz, build_data_matrix, FIELD_CONFIG, POSITION_COLUMNS


def parse_args():
    p = argparse.ArgumentParser(
        description='Run Piecewise (1D) Inversion (PWI) on DUALEM-842s .xyz field data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('xyz_file', help='Raw DUALEM .xyz field data file')
    p.add_argument('-o', '--out-prefix', default=None,
                    help='Output file prefix (default: results/<xyz filename>_pwi)')
    p.add_argument('--nlay', type=int, default=2, help='Number of layers (default: 2)')
    p.add_argument('--height', type=float, default=FIELD_CONFIG.height,
                    help='Instrument height above ground [m] (default: %(default)s)')
    p.add_argument('--lam', type=float, default=0.0,
                    help='Inversion regularisation parameter lambda (default: 0.0, matches the field notebook)')
    p.add_argument('--thk-min', type=float, default=0.1, help='Lower bound for thickness transform [m]')
    p.add_argument('--thk-max', type=float, default=5.0, help='Upper bound for thickness transform [m]')
    p.add_argument('--sig-min', type=float, default=1e-3, help='Lower bound for conductivity transform [S/m]')
    p.add_argument('--sig-max', type=float, default=500e-3, help='Upper bound for conductivity transform [S/m]')
    p.add_argument('--rel-error', type=float, default=1e-3, help='Assumed relative data error')
    return p.parse_args()


def main():
    args = parse_args()
    print(f'Loading  {args.xyz_file}')
    df = read_xyz(args.xyz_file)
    data = build_data_matrix(df)
    npos, n_channels = data.shape

    nlay = args.nlay
    n_par = nlay * 2 - 1
    print(f'{npos} soundings, nlay={nlay}, lam={args.lam}')

    fop = FDEM1DModelling_field(nlay=nlay, height=args.height, config=FIELD_CONFIG)
    fop.region(0).setTransModel(pg.trans.TransLogLU(args.thk_min, args.thk_max))
    fop.region(1).setTransModel(pg.trans.TransLogLU(args.sig_min, args.sig_max))

    start_model = np.ones(n_par)
    start_model[:nlay - 1] *= 3.0   # initial thicknesses [m]
    start_model[nlay - 1:] *= 0.1   # initial conductivities [S/m]

    model_inv = np.full((npos, n_par), np.nan)
    response = np.full((npos, n_channels), np.nan)

    for pos in range(npos):
        dat = data[pos]
        rel_err = np.ones_like(dat) * args.rel_error
        inv = pg.Inversion(fop)
        model_inv[pos] = inv.run(dataVals=dat, errorVals=rel_err,
                                  startModel=start_model, verbose=False, lam=args.lam)
        response[pos] = inv.response
        if (pos + 1) % 100 == 0 or pos + 1 == npos:
            print(f'  inverted {pos + 1}/{npos} soundings')

    if 'LINE' in df.columns:
        for line_id in pd.unique(df['LINE'].to_numpy()):
            idx = np.flatnonzero(df['LINE'].to_numpy() == line_id)
            rmse = np.sqrt(np.mean((data[idx] - response[idx]) ** 2)) * 1000
            print(f'Line {line_id}: {len(idx)} soundings, RMSE = {rmse:.3f} ppt')
    else:
        rmse = np.sqrt(np.mean((data - response) ** 2)) * 1000
        print(f'RMSE = {rmse:.3f} ppt')

    out_prefix = args.out_prefix or os.path.join(
        'results', os.path.splitext(os.path.basename(args.xyz_file))[0] + '_pwi')
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    thk_cols = [f'h_{i + 1}' for i in range(nlay - 1)]
    sig_cols = [f'sigma_{i + 1}' for i in range(nlay)]
    result_df = df[[c for c in POSITION_COLUMNS if c in df.columns]].copy()
    for i, c in enumerate(thk_cols):
        result_df[c] = model_inv[:, i]
    for i, c in enumerate(sig_cols):
        result_df[c + '_mSm'] = model_inv[:, nlay - 1 + i] * 1000

    np.save(out_prefix + '_model.npy', model_inv)
    np.save(out_prefix + '_response.npy', response)
    result_df.to_csv(out_prefix + '.csv', index=False)

    print(f'\nSaved  {out_prefix}.csv')
    print(f'       {out_prefix}_model.npy')
    print(f'       {out_prefix}_response.npy')


if __name__ == '__main__':
    main()
