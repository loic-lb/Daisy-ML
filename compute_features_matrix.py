import argparse
import os
import numpy as np
import pickle

from clustering_analysis import construct_dataset, get_resnet_features

def main():
    parser = argparse.ArgumentParser(description="Compute matrix of ResNet features extracted from patches")
    parser.add_argument("dataset_path", type=str, help="Clinical data file path")
    parser.add_argument("--save_location", type=str, default="./Results", help="Where to save features matrix")
    parser.add_argument("--cohort_indices", type=int, nargs='+', default=[1], help="Cohorts to include in the analysis")

    args = parser.parse_args()

    df, dataset = construct_dataset(args.dataset_path)
    patch_features, _ = get_resnet_features(df, [f"COHORT {cohort_index}" for cohort_index
                                                 in args.cohort_indices], dataset)
    M = np.concatenate(patch_features)
    M_mean = np.mean(M, axis=0)
    M_std = np.std(M, axis=0)
    M_normalized = (M - M_mean) / M_std

    with open(os.path.join(args.save_location, "features_matrix.pickle"), "wb") as fp:
        pickle.dump({"features_matrix": M_normalized, "mean": M_mean, "std": M_std}, fp)


if __name__ == '__main__':
    main()
