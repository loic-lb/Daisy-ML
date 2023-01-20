import os
import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image, ImageOps
from preprocessing import WholeSlideImage, SLIDES_PATH, PATCHES_CLUSTERING_PATH, PATHOLOGIST_ANNOTATIONS_PATH, ROIS_PATH


def init_image(w, h, color_bakground=(255, 255, 255), mask=False):
    """

    :param w: width of image
    :param h: height of image
    :param color_bakground: color of the background
    :param mask: set to True to create a binary mask
    :return: PIL image with the dimension (w,h), in grayscale when mask if set to True
    """
    if mask:
        return np.array(ImageOps.grayscale(Image.new(size=(w, h), mode="RGB", color=0)))
    else:
        return np.array(Image.new(size=(w, h), mode="RGB", color=color_bakground))


def update_canvas(canvas, coord, patch, patch_size, color=None, weighted=False, alpha=0.6):
    """

    :param canvas: image to add the patch
    :param coord: coordinates of the patch on the canvas
    :param patch: image of the patch
    :param patch_size: size of the patch on the canvas
    :param color: color shade to add when weighted
    :param weighted: set to True to add a specific color shade to the patch
    :param alpha: value of the shading
    :return: canvas updated with the patch with an overlapping color (if selected)
    """
    if weighted:
        if color is None:
            print("Please provide a color, choosing red by default")
            color = np.array([255, 0, 0], dtype=np.uint8)
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = cv2.addWeighted(
            np.ones((patch_size[1], patch_size[0], 3), dtype=np.uint8) * color, alpha,
            patch[:patch_size[1], :patch_size[0], :], 1 - alpha, 0, patch[:patch_size[1], :patch_size[0], :])
    else:
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch
    return canvas


def visualise_clusters(slide_name, labels, coords_filtered=None, label=None, downscale_lvl=3):
    """

    :param slide_name: name of the slide
    :param labels: labels of each patch
    :param coords_filtered: specify specific coords of patches among the clustered patches (after attention threshold for instance)
    :param downscale_lvl: downscale level to which observe the clusters
    :param label: plot only patches with matching label
    :return: PIL image of the selected patches from slide_name with overlapping color for each cluster, binary mask, and region of interests coordinates
    """
    colors = plt.cm.Set1(np.linspace(0, 1, max(labels) + 1))[:, :-1] * 255
    WSI_object = WholeSlideImage(os.path.join(SLIDES_PATH, f"{slide_name}.mrxs"), pd.read_csv(ROIS_PATH),
                                 PATHOLOGIST_ANNOTATIONS_PATH)
    wsi = WSI_object.getOpenSlide()
    downscale = wsi.level_downsamples[downscale_lvl]
    vis_level = wsi.get_best_level_for_downsample(downscale)
    file = h5py.File(os.path.join(PATCHES_CLUSTERING_PATH, 'patches', slide_name + ".h5"), 'r')
    dset = file['coords']
    if coords_filtered is None:
        coords = dset[:]
    else:
        coords = coords_filtered
    w, h = wsi.level_dimensions[vis_level]
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    patch_size = tuple((np.array((patch_size, patch_size)) * wsi.level_downsamples[patch_level]).astype(np.int32))
    canvas = init_image(w, h)
    mask = init_image(w, h, mask=True)
    downsamples = WSI_object.wsi.level_downsamples[vis_level]
    idxs = np.arange(len(coords))
    total = len(idxs)
    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downsamples))).astype(np.int32))
    colors = np.array(colors, dtype=np.uint8)
    for idx in range(total):
        patch_id = idxs[idx]
        coord = coords[patch_id]
        patch = np.array(WSI_object.wsi.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        if type(label) == int:
            if labels[idx] == label:
                canvas = update_canvas(canvas, coord, patch, patch_size)
                mask[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = np.ones(
                    (patch_size[1], patch_size[0]), dtype=np.uint8)
        else:
            color = colors[labels[idx]]
            canvas = update_canvas(canvas, coord, patch, patch_size, color, weighted=True)
        # canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[
        # :canvas_crop_shape[0], :canvas_crop_shape[1], :]
    return Image.fromarray(canvas), Image.fromarray(mask), WSI_object.ROIs


def significative_cluster_example(WSI_object, mask, roi_index=0, downscale_lvl=1, color=True, alpha=0.4, beta=0.6,
                                  gamma=0):
    """

    :param WSI_object: input WholeSlideImage object
    :param mask: mask produced by visualise_clusters with chosen cluster
    :param roi_index: region of interest to visualize
    :param downscale_lvl: downscale level to which observe the cluster
    :param color: if set to True, the cluster is highlighted by red shaded patches, else patches not belonging to the cluster
    are darkened
    :param alpha: parameter to control shading
    :param beta: parameter to control shading
    :param gamma: parameter to control shading
    :return: PIL image of the slide within WSI_object highlighting chosen cluster
    """
    wsi = WSI_object.wsi
    ROIs = WSI_object.ROIs
    img = np.asarray(wsi.read_region((0, 0), downscale_lvl, wsi.level_dimensions[downscale_lvl]).convert("RGB").crop(
        ROIs[roi_index][1:] // wsi.level_downsamples[downscale_lvl])).astype(np.uint8)
    if color:
        img[np.where(np.asarray(mask.crop(ROIs[roi_index][1:] // wsi.level_downsamples[downscale_lvl])) == 1)] = alpha * \
                                                                                                                 img[
                                                                                                                     np.where(
                                                                                                                         np.asarray(
                                                                                                                             mask.crop(
                                                                                                                                 ROIs[
                                                                                                                                     roi_index][
                                                                                                                                 1:] //
                                                                                                                                 wsi.level_downsamples[
                                                                                                                                     downscale_lvl])) == 1)] + beta * np.array(
            [255, 0, 0])
        + gamma
    else:
        img[np.where(np.asarray(mask.crop(ROIs[roi_index][1:] // wsi.level_downsamples[downscale_lvl])) == 0)] = beta * \
                                                                                                                 img[
                                                                                                                     np.where(
                                                                                                                         np.asarray(
                                                                                                                             mask.crop(
                                                                                                                                 ROIs[
                                                                                                                                     roi_index][
                                                                                                                                 1:] //
                                                                                                                                 wsi.level_downsamples[
                                                                                                                                     downscale_lvl])) == 0)]
    return Image.fromarray(img)


def patch_closest_to_centroid(labels, WSI_object, kmeans, feats, n_patch=5, downscale_level=1,
                              coords_filtered=None):
    """

    :param labels: cluster assignement of the patches
    :param WSI_object: Whole Slide Image openslide object
    :param kmeans: trained K-Means model
    :param feats: resnet features of the patches
    :param n_patch: number of patch
    :param downscale_level: downscale level to which observe the patches
    :param coords_filtered: specify specific coords of patches among the clustered patches (after attention threshold for instance)
    :return: Images of the closest n_patch to cluster centroids, list of the cluster in WSI_object
    """
    patch_imgs = []
    labels_in_slide = np.unique(labels)
    indices = np.array(range(len(labels)))
    closest = []
    for label in labels_in_slide:
        feats_label = feats[labels == label]
        indices_label = indices[labels == label]
        cluster_center = np.expand_dims(kmeans.cluster_centers_[label], 0)
        closest_label = np.argsort(pairwise_distances(cluster_center, feats_label))[:, :n_patch].flatten()
        closest += list(indices_label[closest_label])
    file = h5py.File(os.path.join(PATCHES_CLUSTERING_PATH, 'patches', WSI_object.name + ".h5"), 'r')
    dset = file['coords']
    if coords_filtered is None:
        coords = dset[:]
    else:
        coords = coords_filtered
    patch_size = dset.attrs['patch_size']
    for patch_idx in closest:
        coord = coords[patch_idx]
        patch_imgs.append(WSI_object.wsi.read_region(tuple(coord), downscale_level, (patch_size, patch_size)))
    return patch_imgs, labels_in_slide


def save_patch_example(save_dir, slide_name, patch_imgs, labels_in_slide, n_clusters, n_patch):
    """

    :param save_dir: save location
    :param slide_name: name of the whole slide image
    :param patch_imgs: list of patches images
    :param labels_in_slide: list of the cluster in the slide
    :param n_clusters: number of clusters
    :param n_patch: number of patch
    """
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(len(labels_in_slide), n_patch),
                     axes_pad=0.2,
                     )
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))[:, :-1]
    n = 0
    for i, (ax, im) in enumerate(zip(grid, patch_imgs)):

        if (i > 0) & (i % n_patch == 0):
            n += 1
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"#{(i % n_patch) + 1}")
        ax.set_ylabel(f"Cluster {labels_in_slide[n]} :", rotation=0, labelpad=40)
        ax.patch.set_edgecolor(colors[labels_in_slide[n]])
        ax.patch.set_linewidth('7')
        ax.imshow(im)

    plt.savefig(os.path.join(save_dir, f"{slide_name}_patches.png"))
    plt.close()
