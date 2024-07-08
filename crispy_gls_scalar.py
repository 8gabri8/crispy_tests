#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from nigsp import io, viz
from nigsp.operations import laplacian
from nigsp.operations.timeseries import resize_ts

FIGSIZE = (50, 25)


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
            "Runs scalar tau estimation with any number of given matrices, either with Ordinary or Generalised Least Squares."
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
            "can be a 1D, txt, csv, tsv, or mat file."
        ),
        required=True,
    )
    required.add_argument(
        "-snn",
        "--input-structural-nonormalisation",
        dest="structural_files_nonorm",
        action="extend",
        nargs="+",
        type=str,
        help=(
            "Complete path (absolute or relative) and name "
            "of the file containing the structural "
            "connectivity matrices. These matrices will NOT be normalised internally. "
            "This file can be a 1D, txt, csv, tsv, or mat file."
        ),
        required=False,
        default=None,
    )

    workflowargs = parser.add_argument_group("Workflow-changing Arguments")

    workflowargs.add_argument(
        "-not0",
        "--no-tau-0",
        dest="add_tau0",
        action="store_false",
        help=("Do not add to the model tau0, associated to I @ ts_{n-1}."),
        default=True,
    )
    workflowargs.add_argument(
        "-nodiff",
        "--no-diff",
        dest="use_diff",
        action="store_false",
        help=("Describe diffusion of Y_t, not of diff(Y_t)"),
        default=True,
    )
    workflowargs.add_argument(
        "-nozs",
        "--no-zscore",
        dest="zscore",
        action="store_false",
        help=("Do not zscore timeseries"),
        default=True,
    )
    workflowargs.add_argument(
        "-nogsr",
        "--no-gsr",
        dest="gsr",
        action="store_false",
        help=("Do not apply column-centering to the timeseries"),
        default=True,
    )
    workflowargs.add_argument(
        "-gls",
        "--general-least-square",
        dest="gls",
        action="store_true",
        help=(
            "After a first pass of OLS, add the covariate matrix of the innovation "
            "signal to improve tau estimation. This will start a recursive estimation, "
            "until either tolerance level or the maximum amount of runs are reached."
        ),
        default=False,
    )
    workflowargs.add_argument(
        "-tol",
        "--tolerance",
        dest="tol",
        type=float,
        help=("Tolerance level. Default 0."),
        default=0,
    )
    workflowargs.add_argument(
        "-max",
        "--max-run",
        dest="max_run",
        type=int,
        help=("Maximum number of reiterations. Default 5000."),
        default=5000,
    )

    optional = parser.add_argument_group("Other Optional Arguments")
    optional.add_argument(
        "-od",
        "--outdir",
        dest="odr",
        type=str,
        help=(
            "Output folder. If None, is specified, a folder `crispy_scalar` will be "
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
    return np.trace(x.T @ y)


def plot_compl(y, norm_time, norm_space, title, filename):
    gs_kw = dict(width_ratios=[0.9, 0.1], height_ratios=[0.2, 0.8])
    yax = np.arange(len(norm_space))
    f, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=gs_kw, figsize=FIGSIZE)
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


def numerical_estimation_scalar_tau(ts_orig, phi, use_diff=True, covmat=None):
    keys = list(phi.keys())

    if covmat is None:
        covmat = np.diag(np.ones(phi[1].shape[0]))
    else:
        covmat = np.linalg.pinv(covmat)

    C = np.empty((len(keys), len(keys)))
    D = np.empty((len(keys)))

    if use_diff:
        print("Initialise models using diff(ts)_t")
        ts_diff = np.diff(ts_orig)
    else:
        print("Initialise models using ts_t")
        ts_diff = ts_orig[:, 1:]

    print("Compute tau_s_ coefficients")
    for n, j in enumerate(keys):
        print(f"Structure of tau {j}")
        D[n] = -tr(ts_diff, covmat @ phi[j])
        for m, k in enumerate(keys):
            C[m, n] = tr(phi[k], covmat @ phi[j])

    print("Estimate tau_s_ through numerical model")
    tau = np.linalg.pinv(C) @ D

    return tau


def innov_estimation_scalar_tau(ts_orig, phi, tau, use_diff=True):
    keys = list(phi.keys())

    print("Reconstruct innovation timeseries")
    if use_diff:
        ts_innov = np.diff(ts_orig)
    else:
        ts_innov = ts_orig[:, 1:]
    for n, k in enumerate(keys):
        ts_innov = ts_innov + tau[n] * phi[k]

    return ts_innov


def multitau_gls_estimation(
    tsfile,
    structural_files,
    structural_files_nonorm=None,
    add_tau0=True,
    use_diff=True,
    zscore=True,
    gsr=True,
    gls=False,
    tol=0,
    max_run=5000,
    odr=None,
    sub="random",
    debug=False,
):
    # ## Check the required variables
    if type(tsfile) is not np.ndarray:
        if not tsfile:
            raise ValueError("Missing timeseries input.")
    structural_files = (
        structural_files if type(structural_files) is list else [structural_files]
    )

    if len(structural_files) < 1:
        raise ValueError("Not enough tau-related matrices declared.")

    if structural_files_nonorm is not None:
        if type(structural_files_nonorm) is not list:
            if type(structural_files_nonorm) is tuple:
                structural_files_nonorm = list(structural_files_nonorm)
            else:
                structural_files_nonorm = [structural_files_nonorm]
        sf_nn_len = len(structural_files_nonorm)
    else:
        structural_files_nonorm = []
        sf_nn_len = 0

    keys = list(range(1, len(structural_files) + sf_nn_len + 1))

    d = dict.fromkeys(keys)
    lapl = dict.fromkeys(keys)
    lapl_norm = dict.fromkeys(keys)
    # ## Read data, transform into Laplacian, and decompose.

    # Check inputs type
    if type(tsfile) is not np.ndarray:
        print("Check input data")
        sc_is = dict.fromkeys(io.EXT_DICT.keys(), "")
        ts_is = dict.fromkeys(io.EXT_DICT.keys(), False)
        sc_nn_is = dict.fromkeys(io.EXT_DICT.keys(), "")
        for k in io.EXT_DICT.keys():
            ts_is[k], _ = io.check_ext(io.EXT_DICT[k], tsfile)
            sc_is[k] = []
            for f in structural_files:
                sc_is[k] += [io.check_ext(io.EXT_DICT[k], f)[0]]
            sc_is[k] = all(sc_is[k])
            if structural_files_nonorm is not None:
                sc_nn_is[k] = []
                for f in structural_files_nonorm:
                    sc_nn_is[k] += [io.check_ext(io.EXT_DICT[k], f)[0]]
                sc_nn_is[k] = all(sc_nn_is[k])

        # Prepare structural connectivity matrix input and read in functional data
        loadfunc = None
        ts = None
        loadfunc_nn = None
        for ftype in io.LOADMAT_DICT.keys():
            if sc_is[ftype]:
                print(f"Structure files will be loaded as an {ftype} file")
                loadfunc = io.LOADMAT_DICT[ftype]
            if sc_nn_is[ftype]:
                print(f"Structure files (skip norm) will be loaded as an {ftype} file")
                loadfunc_nn = io.LOADMAT_DICT[ftype]
            if ts_is[ftype]:
                ts = {}
                print(f"Load {os.path.basename(tsfile)} as an {ftype} file")
                ts["orig"] = io.LOADMAT_DICT[ftype](tsfile)

        if loadfunc is None:
            raise NotImplementedError(
                "Input structural file is not in a supported type."
            )

        if ts is None:
            raise NotImplementedError(
                f"Input file {tsfile} is not of a supported type."
            )
    else:
        ts = {}
        ts["orig"] = tsfile

    # Create folder before it's too late
    # If odr is None, save in the folder of the ts file.
    if odr is None:
        odr = os.path.join(os.path.dirname(tsfile), "crispy_scalar")

    os.makedirs(f"{odr}/plots", exist_ok=True)
    os.makedirs(f"{odr}/files", exist_ok=True)

    # Column-center ts (sort of GSR)
    if zscore:
        print("Z-score timeseries")
        ts["orig"] = resize_ts(ts["orig"], resize="norm")
    if gsr:
        print("Column center timeseries (GSR)")
        ts["orig"] = ts["orig"] - ts["orig"].mean(axis=0)[np.newaxis, ...]

    # ### Create SC matrix
    print("Laplacian-ise the structural files")
    for n, k in enumerate(keys[: len(structural_files)]):
        if type(tsfile) is not np.ndarray:
            d[k] = loadfunc(structural_files[n], shape="square")
        else:
            d[k] = structural_files[n]
        lapl[k], degree = laplacian.compute_laplacian(d[k], selfloops="degree")
        lapl_norm[k] = laplacian.normalisation(lapl[k], degree, norm="rwo")

    # ### Create non-normalised SC matrix
    if sf_nn_len > 0:
        print("Load the structural files without normalisation")
        for n, k in enumerate(keys[-sf_nn_len:]):
            if type(tsfile) is not np.ndarray:
                d[k] = loadfunc_nn(structural_files_nonorm[n], shape="square")
            else:
                d[k] = structural_files_nonorm[n]
            lapl[k] = d[k].copy()
            degree = d[k].sum(axis=1)
            lapl_norm[k] = d[k].copy()

    # ## Compute phi to simplify coefficients
    phi = dict.fromkeys(keys)

    for k in keys:
        phi[k] = lapl_norm[k] @ ts["orig"][:, :-1]

    if add_tau0:
        print("Add t0 to the model")
        keys = [0] + keys
        lapl_norm[0] = np.diag(np.ones(lapl_norm[1].shape[0]))
        phi[0] = ts["orig"][:, :-1].copy()

    tau = numerical_estimation_scalar_tau(ts["orig"], phi, use_diff)
    ts["innov"] = innov_estimation_scalar_tau(ts["orig"], phi, tau, use_diff)

    if gls:
        covmat = {"ols": np.cov(ts["innov"].copy())}
        for n in range(max_run):
            print("Start GLS estimation using previous covariance")
            cov = np.cov(ts["innov"])
            ts_prev = ts["innov"].copy()

            tau = numerical_estimation_scalar_tau(ts["orig"], phi, use_diff, cov)
            ts["innov"] = innov_estimation_scalar_tau(ts["orig"], phi, tau, use_diff)

            t = np.linalg.norm(ts_prev - ts["innov"])

            print(f"Round {n}, tolerance: {t}")
            if t <= tol:
                break
        covmat["gls"] = np.cov(ts["innov"]).copy()
        io.export_txt(np.asarray([n, t]), f"{odr}/sub-{sub}_gls_estimation_log.txt")

    print("Compute norm")
    norm = {}
    for k in ts.keys():
        norm[f"{k}_time"] = np.linalg.norm(ts[k], axis=0)
        norm[f"{k}_space"] = np.linalg.norm(ts[k], axis=-1)

    # ## Plot and export everything

    io.export_txt(tau, f"{odr}/files/sub-{sub}_tau_scalar.tsv")

    # If a previous run created a lapl_norm_0, delete it.
    if not add_tau0 and os.path.exists(f"{odr}/files/sub-{sub}_lapl_norm_0.tsv.gz"):
        print(
            f"Remove existing tau0 file from a previous run: {odr}/files/sub-{sub}_lapl_norm_0.tsv.gz"
        )
        os.remove(f"{odr}/files/sub-{sub}_lapl_norm_0.tsv.gz")

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
            title=f'Timeseries "{k}" sub {sub}, estimated taus: {tau}',
            filename=f"{odr}/plots/sub-{sub}_ts-{k}.png",
        )

    for k in keys:
        io.export_txt(lapl_norm[k], f"{odr}/files/sub-{sub}_lapl_norm_{k}.tsv.gz")

    if gls:
        for k in covmat.keys():
            io.export_txt(
                covmat[k], f"{odr}/files/sub-{sub}_ts-innov_covmat-{k}.tsv.gz"
            )
            viz.plot_connectivity(
                covmat[k], f"{odr}/plots/sub-{sub}_ts-innov_covmat-{k}.png"
            )

    if debug:
        res = {
            "ts": ts,
            "lapl": lapl,
            "lapl_norm": lapl_norm,
            "tau": tau,
        }

        return res


def _main(argv=None):
    options = _get_parser().parse_args(argv)

    multitau_gls_estimation(**vars(options))


if __name__ == "__main__":
    _main(sys.argv[1:])
