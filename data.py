import csv
import h5py
import numpy as np
import scipy.ndimage as nd
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
from pylab import *
import scipy.misc
import tensorflow as tf
import socket
import sys
import time

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
    filenames = listdir(dirname)
    shape = None
    num_channels = 0
    # Trying to look at each image file.
    for filename in filenames:
        filepath = join(dirname, filename)
        if filename[-3:] != '.h5':
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

    raise ValueError("Something went wrong in finding out img dimensions")

def read_data_shape(path):
    """
    Reads an hdf5 file and returns the shape of the 'data' array.
    """
    with h5py.File(path) as hf:
        return hf.get('data').shape

def data_augment(data_iter, data_seg=None, rand_seed=None):
    """
    Stochastically augments the single piece of data.
    INPUT:
    - data_iter: (3d ND-array) the single piece of data
    - data_seg: (2d ND-array) the corresponding segmentation
    """
    matrix_size = data_iter.shape[0]
    # Setting Seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
    # Creating Random variables
    roller = np.round(float(matrix_size/7))
    ox, oy = np.random.randint(-roller, roller+1, 2)
    do_flip = np.random.randn() > 0
    num_rot = np.random.choice(4)
    pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
    add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
    # Rolling
    data_iter = np.roll(np.roll(data_iter, ox, 0), oy, 1)
    if np.any(data_seg):
        data_seg = np.roll(np.roll(data_seg, ox, 0), oy, 1)
    # Left-right Flipping
    if do_flip:
        data_iter = np.fliplr(data_iter)
        if np.any(data_seg):
            data_seg = np.fliplr(data_seg)
    # Random 90 Degree Rotation
    data_iter = np.rot90(data_iter, num_rot)
    if np.any(data_seg):
        data_seg = np.rot90(data_seg, num_rot)
    # Raising/Lowering to a power
    #data_iter = data_iter ** pow_rand
    # Random adding of shade.
    data_iter += add_rand
    if np.any(data_seg):
        return data_iter, data_seg
    return data_iter

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
