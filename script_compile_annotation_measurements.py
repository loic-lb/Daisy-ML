import argparse
import os
import numpy as np
import pandas as pd
import h5py
import pickle
from tqdm import tqdm
from preprocessing import WholeSlideImage as WSI, construct_dataset, SLIDES_PATH, ROIS_PATH, \
    PATHOLOGIST_ANNOTATIONS_PATH, PATCHES_CLUSTERING_PATH
from clustering_analysis import construct_feats


def main():
    parser = argparse.ArgumentParser(description="Compute super-graph from nuclei annotations")
    parser.add_argument('--annotation_measurements', type=str, default="./Results/nuclei/QuPath_project/annotation_measurements",
                        help="path to cell annotations (created with script_extract_nuclei.py)")
    parser.add_argument("--features_matrix", type=str, default="./Results/clustering/features_matrix.pickle",
                        help="Path to features matrix pickle file extracted from the slides")
    parser.add_argument("--pretrained_kmeans", type=str, default="./Results/clustering/kmeans_model.pickle",
                        help="Path to pretrained K-Means model")
    parser.add_argument("--dataset", type=str, default="./dataset.csv", help="Clinical data file path")
    parser.add_argument("--cohort_indices", type=int, nargs='+', default=[1], help="Cohorts to include in the analysis")
    parser.add_argument('--save_location', type=str, default="./Results/nuclei/", help="save location")
    parser.add_argument('--resolution', type=float, default=0.3250, help="resolution in px/um of the slides")

    args = parser.parse_args()

    df, dataset = construct_dataset(args.dataset)

    slides_to_analyse = df[df.cohort.isin([f"COHORT {i}" for i in args.cohort_indices])].slide_id.values

    with open(args.features_matrix, "rb") as input_file:
        features_matrix_dict = pickle.load(input_file)

    with open(args.pretrained_kmeans, "rb") as fp:
        kmeans = pickle.load(fp)

    result = pd.DataFrame()

    for slide_name in tqdm(slides_to_analyse):
        file = h5py.File(os.path.join(PATCHES_CLUSTERING_PATH, 'patches', slide_name + ".h5"), 'r')
        dset = file['coords']
        coords = dset[:]
        patch_size_clustering = (dset.attrs['patch_size'], dset.attrs['patch_size'])
        feats, _ = construct_feats(df, slide_name, dataset)
        feats = (feats - features_matrix_dict["mean"]) / features_matrix_dict["std"]
        labels_clustering = kmeans.predict(feats)

        raw_data = pd.read_csv(os.path.join(args.annotation_measurements, f"{slide_name}.csv"), sep="\t")
        # Filter some noise by minimum perimeter and minimum circularity
        raw_data = raw_data.drop((raw_data[(raw_data["Perimeter µm"] < 8.29) & (raw_data["Circularity"] < 0.31)]).index)
        # Remove useless columns and nan lines
        raw_data = raw_data.drop(columns=["Image", "Name", "Class", "Parent", "ROI", "Object ID"]).dropna()
        # Selecting nuclei centroids
        nuclei = raw_data[["Centroid X µm", "Centroid Y µm"]].values
        raw_data = raw_data.drop(columns=["Centroid X µm", "Centroid Y µm"])
        feats = raw_data.values

        slide_path = os.path.join(SLIDES_PATH, f"{slide_name}.mrxs")
        WSI_object = WSI(slide_path, pd.read_csv(ROIS_PATH), PATHOLOGIST_ANNOTATIONS_PATH)
        xminp, yminp = int(WSI_object.wsi.properties["openslide.bounds-x"]), int(
            WSI_object.wsi.properties["openslide.bounds-y"])

        nuclei_pixels = np.vstack([nuclei[:, 0] / args.resolution + xminp, nuclei[:, 1] / args.resolution + yminp]).T
        for cluster in np.unique(kmeans.labels_):
            patches_in_cluster = coords[labels_clustering == cluster]
            features_patch = []
            for coord_patch in patches_in_cluster:
                nuclei_in_patch = (coord_patch[0] < nuclei_pixels[:, 0]) & (coord_patch[1] < nuclei_pixels[:, 1]) & \
                                  (nuclei_pixels[:, 0] < coord_patch[0] + patch_size_clustering[0]) & (
                                          nuclei_pixels[:, 1] < coord_patch[1] + patch_size_clustering[1])
                if nuclei_in_patch.sum() > 0:
                    features_patch.append(
                        np.array([nuclei_in_patch.sum()] + list(np.mean(feats[nuclei_in_patch], axis=0))))
                else:
                    features_patch.append(np.array([nuclei_in_patch.sum()] + [np.nan] * 12))
            if len(features_patch) == 0:
                continue
            features_cluster = np.vstack(features_patch)
            result = pd.concat(
                [result,
                 pd.DataFrame({"n_nuclei": features_cluster[:, 0],
                               "dab_max_mean": features_cluster[:, 10], "area_mean": features_cluster[:, 1],
                               "length_mean": features_cluster[:, 2],
                               "circularity_mean": features_cluster[:, 3], "solidity_mean": features_cluster[:, 4],
                               "max_diam_mean": features_cluster[:, 5], "min_diam_mean": features_cluster[:, 6],
                               "perimeter_mean": features_cluster[:, 12],
                               "cluster": [cluster] * len(patches_in_cluster),
                               "label": list(df[df.slide_id == slide_name].label.values) * len(patches_in_cluster)})])

    result.to_csv(os.path.join(args.save_location, "nuclei_features.csv"), index=False)


if __name__ == "__main__":
    main()
