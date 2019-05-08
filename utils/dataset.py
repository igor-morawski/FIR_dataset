import argparse
import glob
import os
import pandas as pd

import numpy as np
import cv2

from math import sqrt
import random

import wget
import zipfile


'''
Labels used in
[1] T. Kawashima et al., "Action recognition from extremely low-resolution  
thermal image sequence," 2017 14th IEEE International Conference on Advanced Video
and Signal Based Surveillance (AVSS), Lecce, 2017, pp. 1-6.

'''
PAPER_LABELS_REGEX = dict([
    (r'walk.*', 0),
    (r'sitdown', 1),
    (r'standup', 2),
    (r'falling.*', 3),
    (r'^(sit|lie|stand)$', 4),
])


LABELS_REGEX = dict([
    (r'walk.*', 0),
    (r'sitdown', 1),
    (r'standup', 2),
    (r'falling.*', 3),
    (r'sit', 4),
    (r'lie', 5),
    (r'stand', 6),

])

SKIP_FRAMES = 20
DATSAET_URL = "https://github.com/muralab/Low-Resolution-FIR-Action-Dataset/archive/master.zip"
DATASET_FN_ZIP = "Low-Resolution-FIR-Action-Dataset-master.zip"


def download(dataset_dir: str, dataset_name: str = "dataset"):
    print("Downloading FIR Action Dataset...")
    wget.download(DATSAET_URL, bar=wget.bar_thermometer)
    path, filename = dataset_dir, dataset_name
    with zipfile.ZipFile(DATASET_FN_ZIP, "r") as zip_ref:
        zip_ref.extractall(path)
    os.remove(DATASET_FN_ZIP)
    dataset_fn = DATASET_FN_ZIP.split(".")[-2]
    os.rename(os.path.join(path, dataset_fn), os.path.join(path, filename))
    print("")
    print("Dataset downloaded to %s" % os.path.join(path, filename))
    return


def load_annotation(dataset_dir: str) -> pd.core.frame.DataFrame:
    pattern = os.path.join(dataset_dir, 'annotation', '*_human.csv')
    generator = glob.iglob(pattern)

    return pd.concat([pd.read_csv(fn, header=None)
                      for fn in generator], ignore_index=True)


def read_sequence_annotation(sequence_name: str, annotation: pd.core.frame.DataFrame = None) -> list:
    if annotation is None:
        return []
    sequence_annotation_pd = annotation[annotation[0] == sequence_name]
    return sequence_annotation_pd.iloc[:, 1:].values.tolist()


def list_sequences(dataset_dir: str) -> list:
    pattern = os.path.join(dataset_dir, '*', 'raw', '*.csv')
    generator = glob.iglob(pattern)
    return [sequence for sequence in generator]


def sequence_heatmap(sequence: np.ndarray, min: int = 20, max: int = 40, cv_colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    sequence_clipped = np.clip(sequence, min, max)
    sequence_normalized = (255 * ((sequence_clipped-min) /
                                  (max-min))).astype(np.uint8)
    shape = sequence.shape

    heatmap_flat = cv2.applyColorMap(
        sequence_normalized.flatten(), cv_colormap)

    return heatmap_flat.reshape([shape[0], shape[1], shape[2], 3])


class Dataset():
    def __init__(self, dataset_dir: str, sample: bool = False, samples_k: int = 10, labels=None):
        self.annotation = load_annotation(dataset_dir)
        self.sequences = list_sequences(dataset_dir)
        if sample:
            self.sequences = random.sample(self.sequences, samples_k)
        if labels:
            self.labels = labels
        self.directory = dataset_dir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return Sequence(self.sequences[idx], dataset_annotation=self.annotation)


class Action(Dataset):
    def __init__(self, dataset, label, samples_k=3):
        annotation = dataset.annotation
        self.annotation = annotation[annotation[3].str.contains(
            label)].sample(samples_k)
        #[sequence for sequence in dataset.sequences if b[0].str.contains(sequence.split(os.path.sep)[-1]).any()]
        self.sequences = list(self.annotation[0].unique())
        self.directory = dataset.directory

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        sequence_name = self.annotation[0].iloc[idx]
        fn = os.path.join(self.directory, sequence_name.split("_")[
                          0], "raw", sequence_name)
        return Sequence(fn, frame_start=self.annotation[1].iloc[idx], frame_stop=self.annotation[2].iloc[idx])


class Sequence(np.ndarray):
    def __new__(cls, fn: str, dataset_annotation=None, frame_start=None, frame_stop=None):
        # read dataframe
        dataframe = pd.read_csv(fn, skiprows=[0, 1], header=None)
        # skip time and PTAT columns
        pixels = dataframe.iloc[:, 2:].values
        min = pixels[SKIP_FRAMES:].min()
        max = pixels[SKIP_FRAMES:].max()
        pixels = pixels[frame_start:frame_stop][:]
        # reshape to [frames, h, w] array
        frames, h, w = pixels.shape[0], (int)(
            sqrt(pixels.shape[1])), (int)(sqrt(pixels.shape[1]))
        obj = np.asarray(pixels.reshape([frames, h, w])).view(cls)
        # add custom sequence attributes
        obj.filename = fn
        path, sequence_name = os.path.split(fn)
        obj.sequence_name = sequence_name
        obj.dataset_annotation = dataset_annotation
        obj.start = frame_start
        obj.stop = frame_stop
        obj.temp_min = min
        obj.temp_max = max
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filename = getattr(obj, 'filename', None)
        self.sequence_name = getattr(obj, 'sequence_name', None)
        self.dataset_annotation = getattr(obj, 'dataset_annotation', None)
        self.start = getattr(obj, 'start', None)
        self.stop = getattr(obj, 'stop', None)

    def annotation(self):
        return read_sequence_annotation(self.sequence_name, self.dataset_annotation)
