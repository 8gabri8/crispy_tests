import logging
import matplotlib as plt
import numpy as np

LGR = logging.getLogger(__name__)
SET_DPI = 100
FIGSIZE = (18, 10)

def plot_nodes(ns, atlas, filename=None, thr=None, closeplot=False, node_cmap=plt.cm.viridis_r, display_mode="ortho", figure=None, axes=None, title=None):
    """
    Create a marker plot in the MNI space.

    If ns has 2 dimensions, average first along last dimension.

    Parameters
    ----------
    ns : numpy.ndarray
        A 1- or 2- D array that contains the value of the nodes.
    atlas : str, os.PathLike, 3D Nifti1Image, or numpy.ndarray
        The 3d nifti image of an atlas, a string or path to its position,
        or a list of coordinates of the center of mass of parcels.
    filename : None, str, or os.PathLike, optional
        The path to save the plot on disk.
    thr : float or None, optional
        The threshold to use in plotting the nodes.
    closeplot : bool, optional
        Whether to close plots after saving or not. Mainly used for debug.

    Returns
    -------
    0
        If there are no errors.

    Raises
    ------
    ImportError
        If matplotlib and/or nilearn are not installed.
    ValueError
        If ns has more than 2 dimensions.
        If coordinates can't be extracted from atlas.

    Notes
    -----
    Requires `matplotlib` and `nilearn`
    """
    try:
        import matplotlib.pyplot as plt
        from nilearn.plotting import find_parcellation_cut_coords, plot_markers
    except ImportError:
        raise ImportError(
            "nilearn and matplotlib are required to plot node images. "
            "Please see install instructions."
        )
    # First check that ns is a valid source of data.
    ns = ns.squeeze()
    if ns.ndim > 2:
        raise ValueError("Cannot plot node values for matrix of " "dimensions > 2.")
    elif ns.ndim == 2:
        LGR.warning(
            "Given matrix has 2 dimensions, averaging across last " "dimension."
        )
        ns = ns.mean(axis=-1)

    # Then treat atlas
    if type(atlas) is np.ndarray:
        if atlas.ndim > 2 or atlas.shape[1] != 3:
            raise NotImplementedError(
                "Only atlases in nifti format or " "list of coordinates are supported."
            )
        else:
            coord = atlas
    else:
        coord = find_parcellation_cut_coords(atlas)

    if ns.shape[0] != coord.shape[0]:
        raise ValueError("Node array and coordinates array have different length.")

    LGR.info("Creating markerplot.")
    plt.figure(figsize=FIGSIZE)
    plot_markers(ns, coord, node_threshold=thr, node_cmap=node_cmap, display_mode=display_mode)#, figure=figure, axes=axes, title=title)

    if filename is not None:
        plt.savefig(filename, dpi=SET_DPI)

    if closeplot:
        plt.close()

    return 0



"""
Copyright 2022, Stefano Moia.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""