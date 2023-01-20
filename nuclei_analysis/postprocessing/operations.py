import numpy as np
from skimage import morphology as skmorphology
from scipy import ndimage as nd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def compute_morphological_operations(img, erosion=False):
    img = nd.median_filter(img, size=5)
    if erosion:
        img = skmorphology.binary_erosion(img, skmorphology.disk(2))#5
    img = skmorphology.binary_opening(img, skmorphology.disk(2))
    img = skmorphology.binary_closing(img, skmorphology.disk(2))
    img = skmorphology.binary_opening(img, skmorphology.disk(1))
    img = skmorphology.binary_closing(img, skmorphology.disk(1))
    return img

def compute_watershed(img, sigma=1, disk_size=12):
    distance = nd.morphology.distance_transform_edt(img)
    smoothed_distance = nd.gaussian_filter(distance, sigma=sigma)
    indices = peak_local_max(smoothed_distance, footprint=skmorphology.disk(disk_size),
                                labels=img)
    local_maxi = np.zeros(smoothed_distance.shape, dtype=bool)
    local_maxi[tuple(indices.T)] = True
    markers = nd.label(local_maxi)[0]
    img_post = watershed(-smoothed_distance, markers, mask=img)
    return img_post

