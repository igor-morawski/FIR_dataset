
import argparse
import os
import re

import numpy as np
import cv2

from utils import dataset as fir

from tqdm import tqdm

import imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .csv sequences to videos.")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join('..', 'dataset'), help='Path to FIR dataset')
    parser.add_argument("--output", type=str,
                        default=os.path.join('output'), help='Path to output directory')
    interpolation_methods = {"nearest": cv2.INTER_NEAREST,
                             "linear": cv2.INTER_LINEAR,
                             "cubic": cv2.INTER_CUBIC,
                             "area": cv2.INTER_AREA}
    parser.add_argument("--interpolation", type=str,
                        default="nearest", choices=interpolation_methods.keys(), help='cv2 interpolation method')
    parser.add_argument("--download", action="store_true",
                        help='Process all sequences in the dataset. Otherwise chooses n samples')
    # parser.add_argument("--combine", action="store_true", help='Combine all the gifs together')
    args = parser.parse_args()

    dataset_dir = args.dataset.replace("/", os.sep)
    output_dir = args.output.replace("/", os.sep)
    interpolation_flag = interpolation_methods[args.interpolation]

    if args.download:
        fir.download("..")

    if not os.path.isdir(dataset_dir):
        print("Append flag --download to download the dataset")
        raise OSError(2, 'No such file or directory', dataset_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("Loading in and parsing dataset...")
    dataset = fir.Dataset(dataset_dir)

    labels = tuple(fir.LABELS_REGEX)
    labels_alphanumeric = tuple(re.sub(
        r"[^a-zA-Z]+", "", label) for label in labels)

    actions = [fir.Action(dataset, label) for label in labels]

    [_, h, w] = dataset[0].shape
    zoom = 10
    h_zoomed, w_zoomed = zoom*h, zoom*w
    h_frame, w_frame = (int)(1.1*h_zoomed), w_zoomed

    for action, action_name in tqdm(zip(actions, labels_alphanumeric)):
        for sequence, i in zip(action, range(len(action))):
            fn = action_name + str(i) + '.gif'
            with imageio.get_writer(os.path.join(output_dir, fn), mode='I', duration=0.1) as writer:
                for heatmap in fir.sequence_heatmap(sequence, min=sequence.temp_min, max=sequence.temp_max):
                    heatmap_zoomed = cv2.resize(
                        heatmap, (w_zoomed, h_zoomed), interpolation=interpolation_flag)
                    writer.append_data(heatmap_zoomed[:, :, ::-1])
