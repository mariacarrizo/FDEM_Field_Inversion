"""Laterally Constrained Inversion (LCI) on raw DUALEM-842s .xyz field data.

Groups soundings by their 'LINE' number and inverts each line jointly with
lateral constraints between neighbouring soundings, exactly like
15_LCI_Field.ipynb but reading straight from a raw .xyz export instead of
the pre-built Field_data.npy.

Examples
--------
    python run_lci_field.py survey.xyz
    python run_lci_field.py survey.xyz --nlay 2 --lam 1 --cw2 2
    python run_lci_field.py survey.xyz -o results/ackerdijkse --plot
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pygimli as pg

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'src'))
from FDEM1D import FDEM1DModelling_field, LCModelling

from dualem_xyz import read_xyz, build_data_matrix, FIELD_CONFIG, POSITION_COLUMNS


def parse_args():
    p = argparse.ArgumentParser(
        description='Run Laterally Constrained Inversion (LCI) on DUALEM-842s .xyz field data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('xyz_file', help='Raw DUALEM .xyz field data file')
    p.add_argument('-o', '--out-prefix', default=None,
                    help='Output file prefix (default: results/<xyz filename>_lci)')
    p.add_argument('--nlay', type=int, default=2, help='Number of layers (default: 2)')
    p.add_argument('--height', type=float, default=FIELD_CONFIG.height,
                    help='Instrument height above ground [m] (default: %(default)s)')
    p.add_argument('--lam', type=float, default=1.0,
                    help='Inversion regularisation parameter lambda (default: 1.0, matches the field notebook)')
    p.add_argument('--cw1', type=float, default=1.0,
                    help='Lateral constraint weight for layer thicknesses (default: 1)')
    p.add_argument('--cw2', type=float, default=1.0,
                    help='Lateral constraint weight for conductivities (default: 1)')
    p.add_argument('--thk-min', type=float, default=0.1, help='Lower bound for thickness transform [m]')
    p.add_argument('--thk-max', type=float, default=5.0, help='Upper bound for thickness transform [m]')
    p.add_argument('--sig-min', type=float, default=1e-3, help='Lower bound for conductivity transform [S/m]')
    p.add_argument('--sig-max', type=float, default=500e-3, help='Upper bound for conductivity transform [S/m]')
    p.add_argument('--rel-error', type=float, default=1e-3, help='Assumed relative data error')
    p.add_argument('--plot', action='store_true', help='Save a stitched cross-section PNG per line')
    return p.parse_args()


def invert_line(data_line, nlay, args):
    nM = len(data_line)
    n_par = nlay * 2 - 1

    model_ini = np.ones((nM, n_par))
    model_ini[:, :nlay - 1] *= 3.0   # initial thicknesses [m]
    model_ini[:, nlay - 1:] *= 0.1   # initial conductivities [S/m]

    # nlay/height/config are forwarded by pygimli's LCModelling to every
    # per-sounding FDEM1DModelling_field(...) instance it creates internally.
    LCI = LCModelling(FDEM1DModelling_field, nlay=nlay, height=args.height, config=FIELD_CONFIG)
    LCI.initJacobian(dataVals=data_line, nLay=nlay)
    LCI.createJacobian(model_ini)
    LCI.constraint_matrix(data_line, nLay=nlay)
    LCI.normalization(data_line, cWeight_1=args.cw1, cWeight_2=args.cw2, nLay=nlay)
    LCI.createWeight(data_line, cWeight_1=args.cw1, cWeight_2=args.cw2, nLay=nlay)
    LCI.createConstraints()

    LCI.region(1).setTransModel(pg.trans.TransLogLU(args.thk_min, args.thk_max))
    LCI.region(2).setTransModel(pg.trans.TransLogLU(args.sig_min, args.sig_max))

    data_vector = data_line.ravel()
    relative_error = np.ones_like(data_vector) * args.rel_error

    inv = pg.Inversion(LCI)
    model_inv = inv.run(data_vector, relative_error,
                         startModel=model_ini.ravel(), verbose=False, lam=args.lam)

    model_inv = np.asarray(model_inv).reshape(nM, n_par)
    response = np.asarray(inv.response).reshape(nM, data_line.shape[1])
    return model_inv, response


def plot_line(line_id, model_inv, nlay, elevation, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from showStitched import showStitchedModels

    # showStitchedModels plots raw model values as-is; convert conductivity
    # (stored in S/m) to mS/m to match the cMin/cMax color scale below.
    plot_model = model_inv.copy()
    plot_model[:, nlay - 1:] *= 1000

    fig, ax = plt.subplots()
    showStitchedModels(plot_model, ax=ax, topo=elevation, zMax=10,
                        cMin=10, cMax=200, label='mS/m')
    ax.set_title(f'Line {line_id}')
    ax.set_xlabel('Sounding index')
    ax.set_ylabel('Depth [m]')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'line_{line_id}_lci.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    print(f'Loading  {args.xyz_file}')
    df = read_xyz(args.xyz_file)
    data = build_data_matrix(df)
    n_channels = data.shape[1]

    nlay = args.nlay
    n_par = nlay * 2 - 1
    lines = df['LINE'].to_numpy()
    unique_lines = pd.unique(lines)
    print(f'{len(df)} soundings across {len(unique_lines)} line(s), nlay={nlay}, '
          f'lam={args.lam}, cw1={args.cw1}, cw2={args.cw2}')

    model_inv_full = np.full((len(df), n_par), np.nan)
    response_full = np.full((len(df), n_channels), np.nan)

    for line_id in unique_lines:
        idx = np.flatnonzero(lines == line_id)
        data_line = data[idx]
        if len(idx) < 2:
            print(f'Skipping line {line_id}: only {len(idx)} sounding(s), need >= 2 for LCI')
            continue

        model_inv, response = invert_line(data_line, nlay, args)
        model_inv_full[idx] = model_inv
        response_full[idx] = response

        rmse = np.sqrt(np.mean((data_line - response) ** 2)) * 1000
        print(f'Line {line_id}: inverted {len(idx)} soundings, RMSE = {rmse:.3f} ppt')

        if args.plot:
            elevation = df['GPS_Alt'].to_numpy()[idx] if 'GPS_Alt' in df.columns else np.zeros(len(idx))
            out_dir = (args.out_prefix or 'results/lci') + '_figures'
            png_path = plot_line(line_id, model_inv, nlay, elevation, out_dir)
            print(f'  saved {png_path}')

    out_prefix = args.out_prefix or os.path.join(
        'results', os.path.splitext(os.path.basename(args.xyz_file))[0] + '_lci')
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    thk_cols = [f'h_{i + 1}' for i in range(nlay - 1)]
    sig_cols_Sm = [f'sigma_{i + 1}' for i in range(nlay)]
    result_df = df[[c for c in POSITION_COLUMNS if c in df.columns]].copy()
    for i, c in enumerate(thk_cols):
        result_df[c] = model_inv_full[:, i]
    for i, c in enumerate(sig_cols_Sm):
        result_df[c + '_mSm'] = model_inv_full[:, nlay - 1 + i] * 1000

    np.save(out_prefix + '_model.npy', model_inv_full)
    np.save(out_prefix + '_response.npy', response_full)
    result_df.to_csv(out_prefix + '.csv', index=False)

    print(f'\nSaved  {out_prefix}.csv')
    print(f'       {out_prefix}_model.npy')
    print(f'       {out_prefix}_response.npy')


if __name__ == '__main__':
    main()
