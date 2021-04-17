import csv
import h5py
import numpy as np
import scipy.ndimage as nd
import os
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc
import tensorflow as tf
import socket
import sys
import time
from pyCudaImageWarp import augment3d

def aaFilter(im, factors, slope=1.0 / 3):
    """
        Apply a gaussian anti-aliasing filter, preparing the image to be 
        stretched by the given factors. For example, factors (.5, .5, .5) mean
        the 3D image size will be halved in each dimension.
    """

    # Check inputs
    assert(im.ndim == len(factors))

    # Get the sigmas, exit early if nothing to do
    sigmas = np.maximum(1.0 / np.array(factors)- 1.0, 0.0) * slope
    if np.all(sigmas == 0.0):
        return im

    return nd.filters.gaussian_filter(im, sigmas + 1e-3)

def reduceVoronoi(vor, mask=None):
    """
        Reduce a voronoi diagram to eliminate labels which are not present.
        Ignores negative labels
    """

    # Optionally apply the mask
    if mask is not None:
        vor[~mask] = -1

    # Get the negative labels
    masked = vor < 0

    # Get the voronoi
    objects, inds = np.unique(vor, return_inverse=True)
    voronoi = inds.reshape(vor.shape)
    del inds # To avoid bugs!

    # Shift for negative labels
    numNegative = np.sum(objects < 0)
    assert(np.all(objects[:numNegative] < 0))
    voronoi -= numNegative

    # Count the number of non-negative objects
    num_objects = len(objects) - numNegative

    return voronoi, num_objects
        
def write_nii(path, data):
    """
    Write a Nifti image with a generic header
    """
    import nibabel as nib
        
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, path)

def find_data_shape(dirname, ndim=3):
    """
    Reads in one piece of data to find out number of channels.
    INPUT:
    dirname - (string) path of data
    """
    statement = ''
    shape = None
    num_channels = 0
    # Trying to look at each image file.
    for root, dirs, files in os.walk(dirname):
        for basename in files:
            filepath = os.path.join(root, basename)
            if filepath[-3:] != '.h5':
                continue
            shape = read_data_shape(filepath)
            num_dims = len(shape)
            assert(num_dims >= ndim and num_dims <= ndim + 1)
            if num_dims == ndim:
                num_channels = 1
            elif num_dims == ndim + 1:
                num_channels = shape[ndim]
            else:
                raise ValueError(
                    "Image %s dimensions %d do not match ndim %d" % (
                        filepath, num_dims, ndim))
                    
            return shape, num_channels

    raise ValueError("Failed to read image dimensions")

def read_train_data(path):
    """
    Reads the training data. Returns a list of lists, one for each subdirectory.
    """

    # Get the subdirectories
    dirFiles = []
    for root, dirs, files in os.walk(path):
        for thisDir in dirs:
            dirFiles.append(
                read_dir_files(
                        os.path.join(root, thisDir)
                )
            )

    return dirFiles
        

def read_dir_files(path):
    """
    Read the files in a directory and resolve symlinks. Used for test/train dirs
    """
    return [os.path.realpath( # Follow symlinks
        os.path.join(path, x)
    ) for x in listdir(path) if not x.startswith('.')]


def read_data_shape(path):
    """
    Reads an hdf5 file and returns the shape of the 'data' array.
    """
    with h5py.File(path, 'r') as hf:
        return hf.get('data').shape

"""
    Computes the weighting map, to be used with get_softmax_loss(). This ensures
    that class 0 is balanced in weight with the positive classes. Negative
    classes are removed from the loss function.

    By default, volumes in which all voxels have the same label are assigned
    zero weight. To disable this, set ignore_uniform=False.
        
"""
def get_weight_map(labels, ignore_uniform=True):

    # Count the non-negative labels
    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)

    # Return if there's only one category present
    num_classes = len(uniq)
    if num_classes < 2:
        if ignore_uniform:
            return np.zeros(labels.shape)
        else:
            return np.ones(labels.shape)

    # Assign equal weight to each class, normalized to a total weight of one 
    # per voxel
    num_valid = sum(counts)
    weights = (float(num_valid) / num_classes) / counts

    # Generate the weight map
    weight_map = np.zeros(labels.shape)
    for k in range(len(uniq)):
        weight_map[labels == uniq[k]] = weights[k]

    return weight_map

# Apply a binary mask to a (multi-channel) volume. Sets all voxels outside the 
# mask to -inf
def apply_mask(vol, mask):
    vol[~mask] = -float('inf')
    return vol

def augment_segmentations(segList, api="cuda", device=None):
    """
        Apply small distortions to the list of segmentations, returned as a list of binary masks
    """
    segXforms = [augment3d.get_xform(
        seg, 
        rotMax=(2,) * 3,
        shearMax=(1 + 0.05,) * 3,
        transMax=(5,) * 3,
        otherScale=0.01,
        shape=seg.shape
    ) for seg in segList]
    return augment3d.apply_xforms(
        segXforms, 
        labelsList=[seg > 0 for seg in segList], # Don't want 'ignore' factoring in
        oob_label=0,
        api=api,
        device=device
    )[0] # FIXME Returns a tuple
