import os
import re
import argparse
import pickle
import pandas as pd
import sklearn.metrics
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from clustering_analysis import clusterize


def main():
    parser = argparse.ArgumentParser(description="Compute optimal number of clusters")
    parser.add_argument("--features_matrix", type=str, default="./Results/clustering/features_matrix.pickle",
                        help="Features matrix extracted from the slides")
    parser.add_argument("--save_location", type=str, default="./Results/clustering/", help="Where to save result")
    parser.add_argument("--range", type=int, nargs='+', default=[7, 12], help="Minimal and maximal number of clusters to compute")
    parser.add_argument("--score_to_compute", type=str, choices=["silhouette", "calinski_harabasz", "davies_bouldin"],
                        default="davies_bouldin", help="Score to compute for assessing optimal number"
                                                       "of clusters")
    parser.add_argument("--sample_rate", type=float, default=None, help="Sample rate used to sample features for "
                                                                        "metric computation (advised for silhouette"
                                                                        "score), default = None")

    args = parser.parse_args()
    assert len(args.range) == 2, "Must provide exactly 2 values for range." 
    score = []
    L = range(args.range[0], args.range[1] + 1)
    with open(args.features_matrix, "rb") as input_file:
        features_matrix_dict = pickle.load(input_file)
    M = features_matrix_dict["features_matrix"]
    if args.sample_rate:
        sampler = StratifiedShuffleSplit(1, args.sample_rate, random_state=0)
    metric = getattr(sklearn.metrics, f"{args.score_to_compute}_score")
    for nb_cluster in tqdm(L):
        kmeans = clusterize(M, nb_cluster, True)
        if args.sample_rate:
            to_sample = list(sampler.split(M, kmeans.labels_))[0][1]
            M_sample = M[to_sample, :]
        else:
            M_sample = M
        score.append(metric(M_sample, kmeans.predict(M_sample)))
    df = pd.DataFrame({"nb_clusters": L, f"{args.score_to_compute}_score": score})
    
    cohort_idx = re.search(r'.*(\d).pickle', args.features_matrix).group(1)
    file_name = "score_optimal_nb_clusters_" + args.score_to_compute + f"_cohort{cohort_idx}"
    df.to_csv(os.path.join(args.save_location, file_name + ".csv"), index=False)


if __name__ == '__main__':
    main()
