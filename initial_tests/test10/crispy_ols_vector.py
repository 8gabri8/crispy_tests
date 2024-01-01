#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from nigsp import io
from nigsp.operations.timeseries import resize_ts
from numpy.lib.stride_tricks import sliding_window_view as swv


# Needs nigsp >= 0.17
def _get_parser():
    """
    Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict

    """
    parser = argparse.ArgumentParser(
        description=(
            "Runs vectorial tau estimation with any number of given matrices."
        ),
        add_help=False,
    )
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-f",
        "--input-func",
        dest="tsfile",
        type=str,
        help=(
            "Complete path (absolute or relative) and name "
            "of the file containing fMRI signal."
        ),
        required=True,
    )
    required.add_argument(
        "-s",
        "--input-structural",
        dest="structural_files",
        action="extend",
        nargs="+",
        type=str,
        help=(
            "Complete path (absolute or relative) and name "
            "of the file containing the structural "
            "connectivity matrices. This file "
            "can be a 1D, txt, tsv, or mat file."
        ),
        required=True,
    )

    optional = parser.add_argument_group("Other Optional Arguments")

    optional.add_argument(
        "-not0",
        "--no-tau-0",
        dest="add_tau0",
        action="store_false",
        help=("Do not add to the model tau0, associated to I @ ts_{n-1}."),
        default=True,
    )
    optional.add_argument(
        "-nnl",
        "--no-normal-laplacian",
        dest="normalise_laplacian",
        action="store_false",
        help=("Do not normalise Laplacian matrices."),
        default=True,
    )
    optional.add_argument(
        "-premul",
        "--pre-multiplying-tau",
        dest="premul",
        action="store_true",
        help=("Model tau @ L @ Y rather than L @ tau @ y. Latter is default."),
        default=False,
    )
    optional.add_argument(
        "-od",
        "--outdir",
        dest="odr",
        type=str,
        help=(
            "Output folder. If None, is specified, a folder `crispy_vector` will be "
            "created in the folder of the timeseries file."
        ),
        default=None,
    )
    optional.add_argument(
        "-sub",
        "--subject",
        dest="sub",
        type=str,
        help=("Subject name"),
        default="random",
    )

    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    return parser


# Define some functions to simplify later math
def tr(x, y):
    """Compute Tr(x^T y)."""
    return np.trace(x.T @ y, axis1=-2, axis2=-1)


def plot_compl(y, norm_time, norm_space, title, filename):
    gs_kw = dict(width_ratios=[0.9, 0.1], height_ratios=[0.2, 0.8])
    yax = np.arange(len(norm_space))
    f, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=gs_kw, figsize=(10, 5))
    ax[1, 0].imshow(y, cmap="gray")
    ax[0, 0].plot(norm_time)
    ax[1, 1].plot(norm_space, yax)
    ax[0, 0].set_xlim(0, len(norm_time) - 1)
    ax[1, 1].set_ylim(0, yax[-1])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return


def numerical_estimation_vector_tau(ts_orig, phi, covmat=None):

    # if covmat is None:
    #     covmat = np.diag(np.ones(ts_orig.shape[0]))
    # else:
    #     covmat = np.linalg.inv(covmat)

    C = np.empty((phi.shape[0], phi.shape[0]))

    print("Compute tau_s_ coefficients")
    ts_diff = np.diff(ts_orig)
    D = -tr(ts_diff, phi)
    for n in range(phi.shape[0]):
        print(n)
        C[n, :] = tr(phi[n, ...], phi)

    print("Estimate tau_s_ through numerical model")
    tau = np.linalg.pinv(C) @ D

    return tau


def innov_estimation(ts_orig, phi, tau):
    print("Reconstruct innovation timeseries")
    ts_innov = np.diff(ts_orig)
    for n in range(tau.shape[0]):
        ts_innov = ts_innov + tau[n] * phi[n, ...]

    return ts_innov


def multitau_matrix_estimation(
    tsfile,
    structural_files,
    add_tau0=True,
    normalise_laplacian=True,
    premul=False,
    odr=None,
    sub="random",
):
    # ## Check the required variables
    if not tsfile:
        raise ValueError("Missing timeseries input.")
    structural_files = (
        structural_files if type(structural_files) is list else [structural_files]
    )
    if len(structural_files) < 1:
        raise ValueError("Not enough tau-related matrices declared.")

    # ## Read data, transform into Laplacian, and decompose.
    keys = list(range(1, len(structural_files) + 1))

    mtx = dict.fromkeys(keys)

    # Check inputs type
    print("Check input data")
    sc_is = dict.fromkeys(io.EXT_DICT.keys(), "")
    ts_is = dict.fromkeys(io.EXT_DICT.keys(), False)
    for k in io.EXT_DICT.keys():
        ts_is[k], _ = io.check_ext(io.EXT_DICT[k], tsfile)
        sc_is[k] = []
        for f in structural_files:
            sc_is[k] += [io.check_ext(io.EXT_DICT[k], f)[0]]
        sc_is[k] = all(sc_is[k])

    # Prepare structural connectivity matrix input and read in functional data
    loadfunc = None
    ts = None
    for ftype in io.LOADMAT_DICT.keys():
        if sc_is[ftype]:
            print(f"Structure files will be loaded as an {ftype} file")
            loadfunc = io.LOADMAT_DICT[ftype]
        if ts_is[ftype]:
            ts = {}
            print(f"Load {os.path.basename(tsfile)} as an {ftype} file")
            ts["orig"] = io.LOADMAT_DICT[ftype](tsfile)

    if loadfunc is None:
        raise NotImplementedError("Input structural file is not in a supported type.")

    if ts is None:
        raise NotImplementedError(f"Input file {tsfile} is not of a supported type.")

    # Column-center ts (sort of GSR)
    print("Column center timeseries (GSR)")
    ts["orig"] = resize_ts(ts["orig"], resize="norm")
    ts["orig"] = ts["orig"] - ts["orig"].mean(axis=0)[np.newaxis, ...]

    # ### Create SC matrix
    print("Laplacian-ise the structural files")
    for n, k in enumerate(keys):
        mtx[k] = loadfunc(structural_files[n], shape="square")
        mtx[k][np.diag_indices(mtx[k].shape[0])] = 0
        mtx[k] = np.diag(np.ones(mtx[k].shape[0])) - mtx[k]

    # ## Add tau0 to the model if necessary
    if add_tau0:
        print("Add t0 to the model")
        keys = [0] + keys
        mtx[0] = np.diag(np.ones(mtx[1].shape[0]))

    # ## Compute phi to simplify coefficients
    n_elem = mtx[1].shape[0]
    phi = np.zeros((len(keys) * n_elem,) + ts["orig"][..., :-1].shape)

    if premul:
        Id = np.diag(np.ones(n_elem))

    for n, k in enumerate(keys):
        if premul:
            phi_temp = mtx[k] @ ts["orig"][:, :-1]
        for i in range(n_elem):
            phi[n * n_elem + i, ...] = (
                np.outer(Id[:, i], phi_temp[i, :])
                if premul
                else np.outer(mtx[k][:, i], ts["orig"][i, :-1])
            )
    tau = numerical_estimation_vector_tau(ts["orig"], phi)
    ts["innov"] = innov_estimation(ts["orig"], phi, tau)

    # Reconstruct vector taus
    tau_vec = swv(tau, n_elem)[::n_elem].copy()

    print("Compute norm")
    norm = {}
    for k in ts.keys():
        norm[f"{k}_time"] = np.linalg.norm(ts[k], axis=0)
        norm[f"{k}_space"] = np.linalg.norm(ts[k], axis=-1)

    # ## Plot and export everything
    print("Plot and export everything")

    # If odr is None, save in the folder of the ts file.
    if odr is None:
        odr = os.path.join(os.path.dirname(tsfile), "crispy_vector")

    os.makedirs(f"{odr}/plots", exist_ok=True)
    os.makedirs(f"{odr}/files", exist_ok=True)

    np.savetxt(f"{odr}/files/sub-{sub}_tau_vector.tsv", tau_vec)

    # If a previous run created a mtx_0, delete it.
    if not add_tau0 and os.path.exists(f"{odr}/files/sub-{sub}_mtx_0.tsv.gz"):
        print(
            f"Remove existing tau0 file from a previous run: {odr}/files/sub-{sub}_mtx_0.tsv.gz"
        )
        os.remove(f"{odr}/files/sub-{sub}_mtx_0.tsv.gz")

    print("Save everything")
    for k in ts.keys():
        io.export_txt(ts[k], f"{odr}/files/sub-{sub}_ts-{k}.tsv.gz")
        for nt in ["time", "space"]:
            io.export_txt(
                norm[f"{k}_{nt}"], f"{odr}/files/sub-{sub}_ts-{k}_norm_{nt}.1D"
            )

        plot_compl(
            ts[k],
            norm[f"{k}_time"],
            norm[f"{k}_space"],
            title=f'Timeseries "{k}" sub {sub}',
            filename=f"{odr}/plots/sub-{sub}_ts-{k}.png",
        )

    for k in keys:
        io.export_txt(mtx[k], f"{odr}/files/sub-{sub}_mtx_{k}.tsv.gz")


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    multitau_matrix_estimation(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
