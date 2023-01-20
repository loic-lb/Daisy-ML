import os
import pandas as pd
from .imports import PATCHES_CLUSTERING_PATH
from .datasets import Generic_MIL_Dataset


def construct_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    dataset_resnet = Generic_MIL_Dataset(csv_path=dataset_path,
                                         data_dir=os.path.join(PATCHES_CLUSTERING_PATH, 'feats'),
                                         shuffle=False,
                                         seed=1,
                                         print_info=False,
                                         label_dict={"YES": 0, 'NO': 1},
                                         patient_strat=True,
                                         ignore=[])
    return df, dataset_resnet
