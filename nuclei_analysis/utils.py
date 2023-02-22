import torch
import numpy as np
from geojson import Feature, Polygon
from skimage.measure import find_contours
from nuclei_analysis.models.modules import Generator
from nuclei_analysis.postprocessing import compute_morphological_operations, compute_watershed
from skimage.color import rgb2gray, rgba2rgb
from skimage.util.dtype import dtype_limits


def import_model(model_path, device, norm="instance"):
    """

    :param model_path: path to the unsupervised nuclei segmentation model checkpoint
    :param device: device to run the model on
    :param norm: normalization to use in the model
    :return: the pretrained unsupervised nuclei segmentation model
    """
    model = Generator(norm=norm, use_dropout=False).to(device)
    model.eval()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["generator_ihc_to_mask_state_dict"])
    return model


def get_patch_idx_inside_roi(coords, bbox):
    """

    :param coords: coordinates of the patches
    :param bbox: bounding box of the ROI
    :return: indices of the patches inside the ROI
    """
    valid_idx = []
    l, t, r, b = bbox
    for idx in range(len(coords)):
        x, y = coords[idx]
        if (x >= l) & (r >= x) & (y >= t) & (y <= b):
            valid_idx.append(idx)
    return valid_idx


def extract_nuclei(mask_nuclei, xminp, yminp):
    """

    :param mask_nuclei: mask of the nuclei segmentation
    :param xminp: x coordinate of the top left corner of the slide
    :param yminp: y coordinate of the top left corner of the slide
    :return: list of nuclei coordinates and contours in json format
    """
    coord, mask = mask_nuclei
    watershed_mask = compute_watershed(compute_morphological_operations(mask, erosion=False))
    nuclei = []
    for nucleus_idx in np.unique(watershed_mask):
        if nucleus_idx == 0:
            continue

        canvas = (watershed_mask == nucleus_idx).astype("int")
        if any(canvas[:, 0] + canvas[:, -1] + canvas[0, :] + canvas[-1, :]):
            continue
        nucleus_centroid = [x.mean() for x in canvas.nonzero()]
        nucleus_centroid = [nucleus_centroid[1] + coord[0], nucleus_centroid[0] + coord[1]]

        contours_x, contours_y = find_contours(canvas)[0].T
        nuclei.append(
            [nucleus_centroid, Feature(geometry=Polygon([[(float(coord[0] - xminp + y), float(coord[1] - yminp + x))
                                                          for x, y in zip(contours_x, contours_y)]]))])
    return nuclei


def is_low_contrast(image, fraction_threshold=0.05, lower_percentile=1,
                    upper_percentile=99):
    """Check if the image is low contrast. (see skimage.exposure.is_low_contrast documentation)"""
    image = np.asanyarray(image)

    if image.dtype == bool:
        return not ((image.max() == 1) and (image.min() == 0))

    if image.ndim == 3:

        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    
    return ratio < fraction_threshold


def check_nuclei_in_patch(coord_patch, patch_size, points, xminp, yminp, resolution):
    n = 0
    for i, point in enumerate(points):
        coord_pixels = (point[0] / resolution + xminp, point[1] / resolution + yminp)
        if (coord_patch[0] < coord_pixels[0]) and (coord_patch[1] < coord_pixels[1]) and \
                (coord_pixels[0] < coord_patch[0] + patch_size[0]) and (
                coord_pixels[1] < coord_patch[1] + patch_size[1]):
            n += 1
    return n
