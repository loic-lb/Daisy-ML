import argparse
import os
import numpy as np
import pickle

from clustering_analysis import construct_feats
from preprocessing import construct_dataset


def main():
    parser = argparse.ArgumentParser(description="Compute matrix of ResNet features extracted from patches")
    parser.add_argument("--dataset_path", type=str, default="./dataset.csv", help="Clinical data file path")
    parser.add_argument("--save_location", type=str, default="./Results/clustering/",
                        help="Where to save features matrix")
    parser.add_argument("--cohort_indices", type=int, nargs='+', default=[1], help="Cohorts to include in the analysis")

    args = parser.parse_args()

    df, dataset = construct_dataset(args.dataset_path)

    slides_to_analyse = df[df.cohort.isin([f"COHORT {i}" for i in args.cohort_indices])].slide_id.values

    M = []

    for slide_name in slides_to_analyse:
        M.append(construct_feats(df, slide_name, dataset)[0])
    M = np.concatenate(M)

    M_mean = np.mean(M, axis=0)
    M_std = np.std(M, axis=0)
    M_normalized = (M - M_mean) / M_std
    
    file_name = "features_matrix_cohort"
    for i in args.cohort_indices: file_name += f"_{i}"
    with open(os.path.join(args.save_location, file_name + ".pickle"), "wb") as fp:
        pickle.dump({"features_matrix": M_normalized, "mean": M_mean, "std": M_std}, fp)


if __name__ == '__main__':
    main()
