import numpy as np
from .cluster_utils import construct_feats


def get_features(df_patient, dataset, kmeans, normalize=None, ponderate=False, attention_thresh=None):
    """


    :param df_patient: dataframe of the patient
    :param dataset: dataset containing features matrix of each slide
    :param kmeans: trained kmeans model
    :param normalize: normalize dataset by (mean, std)
    :param ponderate: multiply each patch feature vector by attention score
    :param attention_thresh: minimum score for a patch to be kept (if None all patches are selected)
    :return: cluster percentages average across the slides of the patient
    """
    n_cluster = kmeans.n_clusters
    slidesfeats = []
    for slide_name in df_patient.slide_id.values:
        feats, feats_coords = construct_feats(df_patient, slide_name, dataset, ponderate, attention_thresh)
        if normalize:
            mean, std = normalize
            feats = (feats - mean) / std
        labels = kmeans.predict(feats)
        count = [np.sum(labels == label) / len(labels) for label in range(n_cluster)]
        slidesfeats.append(count)
    return np.mean(slidesfeats, axis=0)
