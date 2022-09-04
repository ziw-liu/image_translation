import cv2
import numpy as np
import scipy.ndimage as ndimage
import sys
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball, binary_opening, binary_erosion

# util functions taken from the translation exercise repo

def hist_clipping(input_image, min_percentile=2, max_percentile=98):

    assert (min_percentile < max_percentile) and max_percentile <= 100

    pmin, pmax = np.percentile(input_image, (min_percentile, max_percentile))
    hist_clipped_image = np.clip(input_image, pmin, pmax)

    return hist_clipped_image


def get_unimodal_threshold(input_image):

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5)),
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, "Error in unimodal thresholding"

    return best_threshold


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(
        np.float32, copy=False
    )  # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            # scale each image individually based on its min and max
            limit = [np.nanmin(im[:]), np.nanmax(im[:])]
        im = (
            (im - limit[0])
            / (limit[1] - limit[0] + sys.float_info.epsilon)
            * (2**bit - 1)
        )
    im = np.clip(
        im, 0, 2**bit - 1
    )  # clip the values to avoid wrap-around by np.astype
    if bit == 8:
        im = im.astype(np.uint8, copy=False)  # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False)  # convert to 16 bit

    return im


def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image
    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


def create_unimodal_mask(input_image, str_elem_size=3, kernel_size=3):

    input_image = im_adjust(
        cv2.GaussianBlur(input_image, (kernel_size, kernel_size), 0)
    )
    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    mask = input_image > thr
    mask = binary_opening(mask, str_elem)
    mask = binary_fill_holes(mask)

    return mask
