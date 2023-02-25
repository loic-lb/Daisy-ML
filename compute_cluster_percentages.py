import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from clustering_analysis import clusterize, get_percentage
from preprocessing import construct_dataset


def main():
    parser = argparse.ArgumentParser(description="Train clustering algorithm and retrieves cluster's percentage")
    parser.add_argument("--features_matrix", type=str, default="./Results/clustering/features_matrix.pickle",
                        help="Path to features matrix pickle file extracted from the slides")
    parser.add_argument("--dataset_path", type=str, default="./dataset.csv", help="Clinical data file path")
    parser.add_argument("--save_location", type=str, default="./Results/clustering/", help="Where to save result")
    parser.add_argument("--cohort_indices", type=int, nargs='+', default=[1], help="Cohorts to include in the analysis")
    parser.add_argument("--nb_clusters", type=int, default=8, help="Number of cluster to segment (ideally chosen "
                                                                   "according to "
                                                                   "optimal_number_clusters.py results)")
    parser.add_argument("--pretrained_kmeans", type=str, default=None, help="Path to pretrained K-Means model")
    args = parser.parse_args()
    with open(args.features_matrix, "rb") as input_file:
        features_matrix_dict = pickle.load(input_file)
    if args.pretrained_kmeans:
        with open(args.pretrained_kmeans, "rb") as fp:
            kmeans = pickle.load(fp)
    else:
        M = features_matrix_dict["features_matrix"]
        kmeans = clusterize(M, args.nb_clusters, True)
    features = []
    df, dataset = construct_dataset(args.dataset_path)
    patient_list = df[df.cohort.isin([f"COHORT {i}" for i in args.cohort_indices])].case_id.unique()
    objective_response_list = df[df.cohort.isin([f"COHORT {i}" for i in args.cohort_indices])].label.values
    for patient_id in tqdm(patient_list):
        df_patient = df[df.case_id == patient_id]
        features.append(get_percentage(df_patient, dataset, kmeans, (features_matrix_dict["mean"],
                                                                     features_matrix_dict["std"])))
    features = np.array(features)
    feats_names = np.array([f"percentage_cluster{str(i)}" for i in range(args.nb_clusters)])
    result = pd.DataFrame(
        {"Feature names": list(feats_names) * len(patient_list), "value": features.flatten(),
         "Objective Response": np.repeat(objective_response_list, features.shape[1])})
    result["Objective Response"] = result["Objective Response"].map(
        {"YES": 'Positive Response', "NO": 'Negative Response'})

    file_name = "percentage_clusters_cohort"
    for i in args.cohort_indices: file_name += f"_{i}"
    if args.pretrained_kmeans:
        file_name += "_pretrained"
    result.to_csv(os.path.join(args.save_location, file_name + ".csv"), index=False)

    if not args.pretrained_kmeans:
        file_name = "kmeans_model_cohort"
        for i in args.cohort_indices: file_name += f"_{i}"
        with open(os.path.join(args.save_location, file_name + ".pickle"), "wb") as fp:
            pickle.dump(kmeans, fp)


if __name__ == '__main__':
    main()
