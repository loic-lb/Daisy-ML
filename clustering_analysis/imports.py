import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from preprocessing import WholeSlideImage

ATTENTION_WEIGHTS_PATH = "/"  # not used here
SLIDE_PATH = "/path/to/wholeslideimages/"
ROIS_PATH = "/path/to/roisdf"
PATHOLOGIST_ANNOTATIONS_PATH = "/path/to/annotations"
PATCHES_PATH = "./Results"

__all__ = ['os', 'h5py', 'np','pd', 'Image', 'ImageOps', 'tqdm', 'WholeSlideImage',
           'ATTENTION_WEIGHTS_PATH', 'SLIDE_PATH', 'ROIS_PATH',
           'PATHOLOGIST_ANNOTATIONS_PATH', 'PATCHES_PATH']

