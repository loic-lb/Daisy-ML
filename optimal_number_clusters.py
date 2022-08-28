import os
import argparse
import pickle
import pandas as pd
import sklearn.metrics
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from clustering_analysis.clustering import clusterize


def main():
    parser = argparse.ArgumentParser(description="Compute optimal number of clusters")
    parser.add_argument("features_matrix", type=str, help="Features matrix extracted from the slides")
    parser.add_argument("--save_location", type=str, default="./Results", help="Where to save result")
    parser.add_argument("--range_max", type=int, default=15, help="Maximal number of clusters to test")
    parser.add_argument("--score_to_compute", type=str, choices=["silhouette, calinski, davies_bouldin"],
                        default="davies_bouldin", help="Score to compute for assessing optimal number"
                                                       "of clusters")
    parser.add_argument("--sample_rate", type=float, default=None, help="Sample rate used to sample features for "
                                                                        "metric computation (advised for silhouette"
                                                                        "score), default = None")

    args = parser.parse_args()
    score = []
    L = range(2, args.range_max + 1)
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
    df = pd.DataFrame({"nb_cluster": L, f"{args.score_to_compute}_score": score})
    df.to_csv(os.path.join(args.save_location, "score_optimal_nb_clusters.csv"), index=False)


if __name__ == '__main__':
    main()
