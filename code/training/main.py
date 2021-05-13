import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path
import torch
from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
from matplotlib import pyplot as plt
import os, random, gc
import re, time, json
from  ast import literal_eval
from IPython.display import Audio
from sklearn.metrics import label_ranking_average_precision_score
from tqdm.notebook import tqdm
import joblib

from config import GlobalConfig
from utils.logger import log
from model.model_zoo import BirdClefModel
from data.dataset import prepare_loader, BirdClefDataset

def load_data(df, config):
    def load_row(row, config):
        impath = config.TRAIN_IMAGE_PATH/f"{row.primary_label}/{row.filename}.npy"
        return row.filename, np.load(str(impath))[:config.MAX_READ_SAMPLES]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row(config))
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res


config = GlobalConfig
df = pd.read_csv(config.CSV_PATH, nrows=None)
df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)
LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_image_store = load_data(df)

#initialise logger
logger = log(config, 'training')
logger.info(config.__dict__)
logger.info("-------------------------------------------------------------------")

fold = 0
model = BirdClefModel(config.MODEL_NAME, config.NUM_CLASSES)
train_df = df[df["fold"] != fold].reset_index(drop=True)
valid_df = df[df["fold"] == fold].reset_index(drop=True)

train_data = BirdClefDataset(audio_image_store, train_df, num_classes=config.NUM_CLASSES,
                           sr=config.SR, duration=config.DURATION, is_train=True)
valid_data = BirdClefDataset(audio_image_store, valid_df, num_classes=config.NUM_CLASSES,
                            sr=config.SR, duration=config.DURATION, is_train=False)

train_loader, valid_loader = prepare_loader(train_data, valid_data, config)
fitter = Fitter(model, device, config)
fitter.fit(train_loader, valid_loader, fold)
