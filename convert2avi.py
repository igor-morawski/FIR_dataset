'''
Convert .csv sequences in FIR dataset to videos (.avi) and save in the output path. 
By default converts only 10 randomly chosen samples and saves using %03d.avi pattern.
Run:
    python convert2avi.py --samples=n
to specify number n of samples chosen.
Run:
    python convert2avi.py --all
to process all samples and save using sequence's name.
Run:
    python convert2avi.py --help
to get list of all the program arguments.
'''
import argparse
import os

import numpy as np
import cv2

from utils import dataset as fir

from tqdm import tqdm


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
                        help='Download the dataset.')
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--samples", type=int, default=10,
                        help='Number of sequences chosen randomly(if not --all) from dataset to visualize. Stored in %%03d.avi format. If --all read-don\'t-care')
    parser.add_argument("--gamma", type=float, default=None,
                        help='Gamma correction.')
    args = parser.parse_args()

    print(__doc__)

    dataset_dir = args.dataset.replace("/", os.sep)
    output_dir = args.output.replace("/", os.sep)
    interpolation_flag = interpolation_methods[args.interpolation]
    sample_flag = not args.all
    gamma = args.gamma

    if gamma:
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    if args.download:
        fir.download("..")

    if not os.path.isdir(dataset_dir):
        print("Append flag --download to download the dataset")
        raise OSError(2, 'No such file or directory', dataset_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("Loading in and parsing dataset...")
    dataset = fir.Dataset(dataset_dir, sample=sample_flag,
                          samples_k=args.samples)

    [_, h, w] = dataset[0].shape
    zoom = 100
    h_zoomed, w_zoomed = zoom*h, zoom*w
    h_frame, w_frame = (int)(1.1*h_zoomed), w_zoomed

    x_text, y_text = 0, h_frame - 30
    x_text_minmax = (int)(w_zoomed/2)
    y_text_min = h_frame - 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4

    fn_idx = 0

    for sequence in tqdm(dataset):
        if sample_flag:
            fn = "%03d" % fn_idx + ".avi"
            fn_idx += 1
        else:
            fn = sequence.sequence_name[:-4] + ".avi"
        video = cv2.VideoWriter(os.path.join(output_dir, fn), cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (w_frame, h_frame))
        # initialize annotation for the whole sequence
        labels = list()
        for _ in range(len(sequence)):
            labels.append(None)
        # parse annotation for a sequence
        for action in sequence.annotation():
            for idx in range(action[0], action[1]+1):
                labels[idx] = action[2]
        frame_minmax = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
        min = sequence[fir.SKIP_FRAMES:].min()
        max = sequence[fir.SKIP_FRAMES:].max()
        cv2.putText(frame_minmax, "min: %.2f" % min, (x_text_minmax, y_text_min), font,
                    font_scale/2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_minmax, "max: %.2f" % max, (x_text_minmax, y_text), font,
                    font_scale/2, (255, 255, 255), 2, cv2.LINE_AA)
        if gamma:
            sequence = np.clip(sequence, min, max)
            sequence = (255*(sequence-min) /
                        (max-min)).astype(np.uint8)
            min = 0
            max = 255
            sequence = cv2.LUT(sequence, lookUpTable)
        PTAT_compensate = True
        if PTAT_compensate:
            sequence = sequence.reshape(
                [sequence.shape[0], -1]) - sequence.PTAT
            sequence = sequence.reshape([sequence.shape[0], 16, 16])
            min = 0
            max = sequence[fir.SKIP_FRAMES:].max()
        for (heatmap, label) in zip(fir.sequence_heatmap(sequence, min, max), labels):
            heatmap_zoomed = cv2.resize(
                heatmap, (w_zoomed, h_zoomed), interpolation=interpolation_flag)
            frame = frame_minmax.copy()
            frame[:h_zoomed, :, :] = heatmap_zoomed
            if label:
                cv2.putText(frame, label, (x_text, y_text), font,
                            font_scale, (255, 255, 255), font_scale, cv2.LINE_AA)

            video.write(frame)
