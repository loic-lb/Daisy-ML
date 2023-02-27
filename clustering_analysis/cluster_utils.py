import os
import h5py
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from preprocessing import to_percentiles, ATTENTION_WEIGHTS_PATH, PATCHES_CLUSTERING_PATH


def clusterize(M, n_clusters=2, minibatch=False):
    """

    :param M: features matrix
    :param n_clusters: number of clusters
    :param minibatch: perform minibatch clustering (when M size too big)
    :return: K-Means model trained on M
    """
    if minibatch:
        kmeans = MiniBatchKMeans(
            init="k-means++",
            n_clusters=n_clusters,
            batch_size=1000,
            n_init=500,
            max_no_improvement=100,
            verbose=0,
            random_state=1).fit(M)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=0).fit(M)
    return kmeans


def get_attention_coords_score(df, slide_name):
    """

    :param df: dataframe of the dataset
    :param slide_name: slide index of the selected slide
    :return: coordinates of patches' attention map and corresponding scores
    """
    grouping = df[df.slide_id == slide_name].IHC.values[0]
    r_slide_save_path = os.path.join(ATTENTION_WEIGHTS_PATH, grouping, slide_name)
    block_map_save_path = os.path.join(r_slide_save_path, '{}_blockmap.h5'.format(slide_name))
    file = h5py.File(block_map_save_path, 'r')
    dset = file['attention_scores']
    coord_dset = file['coords']
    scores = dset[:]
    coords = coord_dset[:]
    scores = to_percentiles(scores)
    scores /= 100
    return coords, scores


def get_clustering_coords(slide_name):
    """

    :param slide_name: name of the slide
    :return: coordinates of the patches used for clustering
    """
    hdf5_file_path = os.path.join(PATCHES_CLUSTERING_PATH, 'patches', slide_name + ".h5")
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    return coords


def adapt_attention_to_cluster_coords(attention_coords, cluster_coords, scores, patch_size_big=256,
                                      patch_size_small=64):
    """

    :param attention_coords: coordinates of the patches of the attention map
    :param cluster_coords: coordinates of the patches used for clustering
    :param scores: scores of the patches of the attention map
    :param patch_size_big: size of the patches from the attention map
    :param patch_size_small: size of the patches from the clustering
    :return: indices in the dataset, coordinates and scores of the patches used for clustering.
    The patches have been selected if they belong to a patch from the attention map and have been attributed
    the corresponding attention score
    """
    new_indices = []
    new_coords = []
    new_scores = []
    for i, cd in enumerate(attention_coords):
        coords_filtered = (cluster_coords[:, 0] >= cd[0]) & (cluster_coords[:, 1] >= cd[1]) & (
                cluster_coords[:, 0] + patch_size_small <= cd[0] + patch_size_big) & (
                                  cluster_coords[:, 1] + patch_size_small <= cd[1] + patch_size_big)
        coords_idx = np.where(coords_filtered)[0]
        new_indices.extend(list(coords_idx))
        new_coords.extend(list(cluster_coords[coords_filtered]))
        new_scores.extend([scores[i]] * len(coords_idx))
    new_indices = np.array(new_indices)
    new_coords = np.array(new_coords)
    new_scores = np.array(new_scores)
    return new_indices, new_coords, new_scores


def get_attention_filtered_resnet_feats(dataset, slide_idx, indices):
    """

    :param dataset: dataset with features of the patches
    :param slide_idx: slide index of the selected slide in the dataset
    :param indices: indices in the dataset of the selected patches for this slide
    :return: array of features from the selected patches
    """
    return np.array(dataset[slide_idx][0])[indices]


def construct_feats(df, slide_name, dataset, ponderate=None, attention_thresh=None):
    """

    :param df: dataframe with slide indices and IHC values
    :param slide_name: name of the slide
    :param dataset: dataset containing features matrix of each slide
    :param ponderate: multiply each patch feature vector by attention score
    :param attention_thresh: minimum score for a patch to be kept (if None all patches are selected)
    :return: feature vector, coordinates vector
    """
    slide_idx = df[df.slide_id == slide_name].index[0]
    if None in (attention_thresh, ponderate):
        feats = np.array(dataset[slide_idx][0])
        feats_coords = None
    else:
        attention_coords, scores = get_attention_coords_score(df, slide_name)
        cluster_coords = get_clustering_coords(slide_name)
        feats_indices, feats_coords, feats_scores = adapt_attention_to_cluster_coords(attention_coords,
                                                                                      cluster_coords, scores)
        if ponderate:
            feats = get_attention_filtered_resnet_feats(dataset, slide_idx, feats_indices)
            return feats * feats_scores.reshape(-1, 1), feats_coords
        elif attention_thresh:
            feats_indices = feats_indices[np.where(feats_scores > attention_thresh)[0]]
            feats_coords = feats_coords[np.where(feats_scores > attention_thresh)[0]]
            feats = get_attention_filtered_resnet_feats(dataset, slide_idx, feats_indices)
    return feats, feats_coords


def get_percentage(df_patient, dataset, kmeans, normalize=None):
    """


    :param df_patient: dataframe of the patient
    :param dataset: dataset containing features matrix of each slide
    :param kmeans: trained kmeans model
    :param normalize: normalize dataset by (mean, std)
    :return: cluster percentages average across the slides of the patient
    """
    n_cluster = kmeans.n_clusters
    slidesfeats = []
    for slide_name in df_patient.slide_id.values:
        feats, _ = construct_feats(df_patient, slide_name, dataset)
        if normalize:
            mean, std = normalize
            feats = (feats - mean) / std
        labels = kmeans.predict(feats)
        count = [np.sum(labels == label) / len(labels) for label in range(n_cluster)]
        slidesfeats.append(count)
    return np.mean(slidesfeats, axis=0)
