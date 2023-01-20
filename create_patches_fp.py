# internal imports
from preprocessing import WholeSlideImage, StitchCoords, initialize_df
# other imports
import os
import time
import argparse
import pandas as pd


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, area_min=filter_params["area_min"])

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, ROIs, pathologist_annotations,
                  patch_size=256, step_size=256,
                  seg_params={"seg_level": 0, "window_avg": 3, "window_eng": 3, "thresh": 90},
                  filter_params={'area_min': 2e3},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level=0,
                  use_default_params=False,
                  seg=False, save_mask=True,
                  stitch=False,
                  patch=False, auto_skip=True, process_list=None):
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path, pd.read_csv(ROIs), pathologist_annotations)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
        if len(WSI_object.contours_tissue) == 0:
            print("No masked saved passed because no contours")
            print("patching and stiching passed because no contours")
            df.loc[idx, 'status'] = 'No tumor detected'
        else:
            if save_mask:
                mask = WSI_object.visWSI(**current_vis_params)
                mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
                mask.save(mask_path)

            patch_time_elapsed = -1  # Default time
            if patch:
                current_patch_params.update(
                    {'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                     'save_path': patch_save_dir})
                file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

            stitch_time_elapsed = -1
            if stitch:
                file_path = os.path.join(patch_save_dir, slide_id + '.h5')
                if os.path.isfile(file_path):
                    heatmap, stitch_time_elapsed = stitching(file_path, WSI_object,
                                                             downscale=WSI_object.level_downsamples[
                                                                 current_vis_params['vis_level']])
                    stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                    heatmap.save(stitch_path)

            print("segmentation took {} seconds".format(seg_time_elapsed))
            print("patching took {} seconds".format(patch_time_elapsed))
            print("stitching took {} seconds".format(stitch_time_elapsed))
            df.loc[idx, 'status'] = 'processed'

            seg_times += seg_time_elapsed
            patch_times += patch_time_elapsed
            stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('source', type=str,
                    help='path to folder containing raw wsi image files')
parser.add_argument('ROIs', type=str,
                    help='path to file containing ROIs')
parser.add_argument('--pathologist_annotations', type=str, default="",
                    help='path to folder containing tumor areas masks (when available)')
parser.add_argument('--step_size', type=int, default=256,
                    help='step_size')
parser.add_argument('--patch_size', type=int, default=256,
                    help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type=str, default="./Results",
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
                    help='downsample level at which to patch')
parser.add_argument('--process_list', type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None

    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)

    directories = {'source': args.source,
                   'ROIs': args.ROIs,
                   'pathologist_annotations': args.pathologist_annotations,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source', 'ROIs']:
            os.makedirs(val, exist_ok=True)

    seg_params = {"seg_level": 3, "window_avg": 30, "window_eng": 3, "thresh": 20}
    filter_params = {'area_min': 1.5e3}
    vis_params = {'vis_level': 3, 'line_thickness': 30}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size=args.patch_size, step_size=args.step_size,
                                           seg=args.seg, use_default_params=False, save_mask=True,
                                           stitch=args.stitch,
                                           patch_level=args.patch_level, patch=args.patch,
                                           process_list=process_list, auto_skip=args.no_auto_skip)
