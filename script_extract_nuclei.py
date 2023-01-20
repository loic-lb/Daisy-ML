import os
import argparse
import h5py
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from geojson import dump
from preprocessing import WholeSlideImage as WSI
import functools
from scipy.spatial import distance

from nuclei_analysis import import_model, get_patch_idx_inside_roi, is_low_contrast, extract_nuclei
from preprocessing import construct_dataset, SLIDES_PATH, PATCHES_NUCLEI_PATH, ROIS_PATH, PATHOLOGIST_ANNOTATIONS_PATH

def main():
    parser = argparse.ArgumentParser(description="Extract nuclei segmentation from WSI using unsupervised trained model")
    parser.add_argument("--dataset_path", type=str, default="./dataset.csv", help="Clinical data file path")
    parser.add_argument("--correction_groups", type=str, default="./correction_groups.csv",
                        help="path to csv file with slides' groups for type of correction needed")
    parser.add_argument("--cohort_indices", type=int, nargs='+', default=[1], help="Cohorts to include in the analysis")
    parser.add_argument('--pathologist_annotations', type=str, default="./",
                        help='path to folder containing pathologist\'s annotations (when available)')
    parser.add_argument('--save_location', type=str, default="./Results/nuclei/nuclei_annotations",
                        help="path to save location")
    parser.add_argument('--model_path', type=str, default="./model_daisy_HER2.pt", help="path to the model (pt file)")

    args = parser.parse_args()

    Path(args.save_location).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = import_model(args.model_path, device)

    transform_patch = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    pool = Pool(os.cpu_count() - 1)

    df, _ = construct_dataset(args.dataset_path)
    
    slides_to_analyse = df[df.cohort.isin([f"COHORT {i}" for i in args.cohort_indices])].slide_id.values
        
    slides_to_analyse = [os.path.join(SLIDES_PATH, f"{slide}.mrxs") for slide in slides_to_analyse]
    
    
    correction_groups = pd.read_csv(args.correction_groups)

    for slide_path in slides_to_analyse:

        masks = []
        WSI_object = WSI(slide_path, pd.read_csv(ROIS_PATH), PATHOLOGIST_ANNOTATIONS_PATH)

        xminp, yminp = int(WSI_object.wsi.properties["openslide.bounds-x"]), int(
            WSI_object.wsi.properties["openslide.bounds-y"])
        patch_file = h5py.File(os.path.join(PATCHES_NUCLEI_PATH, "patches", f"{WSI_object.name}.h5"))
        dset = patch_file['coords']
        coords = dset[:]
        print(coords.shape)
        patch_size = dset.attrs['patch_size']
        patch_level = dset.attrs['patch_level']
        idxs = np.arange(len(coords))
                
        contrast_aug_lvl = 1.2 if correction_groups[correction_groups.slide_id==WSI_object.name].group.values[0] == 1 else 1.4
        percentiles = (1, 99) if correction_groups[correction_groups.slide_id==WSI_object.name].group.values[0] == 1 else (10, 90)

        print(f"Working on {WSI_object.name} ...")

        for roi_index, roi in enumerate(WSI_object.ROIs):

            l, t, r, b = roi[1:]
            valid_idx = get_patch_idx_inside_roi(coords, [l, t, r, b])

            for idx in tqdm(valid_idx):
                patch_id = idxs[idx]
                coord = coords[patch_id]
                patch = WSI_object.wsi.read_region(tuple(coord), patch_level, (patch_size, patch_size)).convert(
                    "RGB")

                if is_low_contrast(patch, 0.18, percentiles[0], percentiles[1]):
                    v = torch.arange(1.0, contrast_aug_lvl, 0.1)
                    img_torch = transforms.ToTensor()(patch)
                    batch_aug = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                        torch.clip(torch.outer(v, img_torch.ravel()).reshape([len(v)] + list(img_torch.shape)), 0, 1))
                    mask_nuclei = (model(batch_aug.to(device)) > 0.5).float().cpu()
                    mask_nuclei = torch.min(mask_nuclei, 0).values.squeeze(0).numpy()
                else:
                    img_torch = transform_patch(patch)
                    mask_nuclei = (model(img_torch.unsqueeze(0).to(device)) > 0.5).float().cpu().squeeze(0).squeeze(
                        0).numpy()
                masks.append((coord, mask_nuclei))

        f = functools.partial(extract_nuclei, xminp=xminp, yminp=yminp)
        all_nuclei = pool.map(f, masks)
        all_nuclei = np.array([nucleus for patch_nuclei in all_nuclei for nucleus in patch_nuclei], dtype="object")
        all_nuclei_centroid = np.vstack(all_nuclei[:, 0])
        valid_nuclei = []
        print(len(all_nuclei))
        for i in range(len(all_nuclei)):
            if len(valid_nuclei) < 1:
                valid_nuclei.append(i)
            else:
                if i % 10000 == 0:
                    print(i)
                nucleus_centroid, nucleus_boundaries = all_nuclei[i]
                if distance.cdist([nucleus_centroid], all_nuclei_centroid[valid_nuclei]).min() > 20:
                    valid_nuclei.append(i)
        with open(os.path.join(args.save_location, f'nuclei_{WSI_object.name}.json'), 'w') as f:
            dump(list(all_nuclei[valid_nuclei][:, 1]), f)
    pool.close()


if __name__ == '__main__':
    main()
