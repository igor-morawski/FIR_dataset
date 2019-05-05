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


def read_sequence_annotation(sequencename: str, annotation: pd.core.frame.DataFrame = None) -> list:
    if annotation is None:
        return []
    sequence_annotation_pd = annotation[annotation[0] == sequencename]
    return sequence_annotation_pd.iloc[:, 1:].values.tolist()


def list_sequences(dataset_dir: str) -> list:
    pattern = os.path.join(dataset_dir, '*', 'raw', '*.csv')
    generator = glob.iglob(pattern)
    return [sequence for sequence in generator]


def sequence_heatmap(sequence: np.ndarray, min: int = 20, max: int = 40, cv_colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    sequence_clipped = np.clip(sequence, min, max)
    sequence_normalized = (255*(sequence_clipped-min) /
                           (max-min)).astype(np.uint8)
    shape = sequence.shape

    heatmap_flat = cv2.applyColorMap(
        sequence_normalized.flatten(), cv_colormap)

    return heatmap_flat.reshape([shape[0], shape[1], shape[2], 3])


class Dataset():
    def __init__(self, dataset_dir: str, sample: bool = False, samples_k: int = 10):
        self.annotation = load_annotation(dataset_dir)
        self.sequences = list_sequences(dataset_dir)
        if sample:
            self.sequences = random.sample(self.sequences, samples_k)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return Sequence(self.sequences[idx], dataset_annotation=self.annotation)


class Sequence(np.ndarray):
    def __new__(cls, fn: str, dataset_annotation=None):
        # read dataframe
        dataframe = pd.read_csv(fn, skiprows=[0, 1], header=None)
        # skip time and PTAT columns
        pixels = dataframe.iloc[:, 2:].values
        # reshape to [frames, h, w] array
        frames, h, w = pixels.shape[0], (int)(
            sqrt(pixels.shape[1])), (int)(sqrt(pixels.shape[1]))
        obj = np.asarray(pixels.reshape([frames, h, w])).view(cls)
        # add custom sequence attributes
        obj.filename = fn
        path, sequencename = os.path.split(fn)
        obj.sequencename = sequencename
        obj.dataset_annotation = dataset_annotation
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.filename = getattr(obj, 'filename', None)
        self.sequencename = getattr(obj, 'sequencename', None)
        self.dataset_annotation = getattr(obj, 'dataset_annotation', None)

    def annotation(self):
        return read_sequence_annotation(self.sequencename, self.dataset_annotation)
