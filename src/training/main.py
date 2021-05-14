import sys
sys.path.append("../../src/")

import joblib
import logging
import argparse
import os, random, gc
import re, time, json
import pandas as pd
import numpy as np
from pathlib import Path
from  ast import literal_eval
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import torch
from torch import nn, optim

#Audio libraries
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
from  IPython.display import Audio

#Local .py
from config import GlobalConfig
from utils.logger import log
from model.model_zoo import BirdClefModel
from data.dataset import prepare_loader, BirdClefDataset
from engine import Fitter



#Load spectrogram images to memory
def load_data(df, config):
    def load_row(row):
        impath = config.TRAIN_IMAGE_PATH/f"{row.primary_label}/{row.filename}.npy"
        return row.filename, np.load(str(impath))[:config.MAX_READ_SAMPLES]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res

#Set seed for reproducibility
def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

#Plot train/validation history
def plot_history(train, valid, fold, config):
    epochs = [i for i in range(config.NUM_EPOCHS)]
    plt.subplots(figsize=(12, 10))
    plt.tight_layout
    for idx, key in enumerate(train.keys()):
        plt.subplot(3,2,idx+1)
        plt.plot(epochs, train[key], color='c')
        plt.plot(epochs, valid[key], color='orange')
        plt.ylabel(key)
        plt.legend(['train', 'valid'])

    plt.savefig(config.SAVE_PATH/f'{config.MODEL_NAME}_history_fold_{fold}.png')


 #Train single fold
def train_fold(df, config, device, fold, audio_image_store, logger):
    model = BirdClefModel(config.MODEL_NAME, config.NUM_CLASSES).to(device)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    #log fold statistics
    logger.info("Fold {}: Number of unique labels in train: {}".format(fold, train_df['primary_label'].nunique()))

    train_data = BirdClefDataset(audio_image_store, train_df, num_classes=config.NUM_CLASSES,
                              sr=config.SR, duration=config.DURATION, config=config, is_train=True)
    valid_data = BirdClefDataset(audio_image_store, valid_df, num_classes=config.NUM_CLASSES,
                              sr=config.SR, duration=config.DURATION, config=config, is_train=False)
    train_loader, valid_loader = prepare_loader(train_data, valid_data, config)

    fitter = Fitter(model, device, config)
    train_tracker, valid_tracker = fitter.fit(train_loader, valid_loader, fold)
    plot_history(train_tracker, valid_tracker, fold, config)



#Main Loop
def train_loop(df, config, device, audio_image_store, logger, fold_num:int=None, train_one_fold=False):
    if train_one_fold:
        train_fold(df, config, device, fold_num, audio_image_store, logger)
    else:
        for fold in range(config.NUM_FOLDS):
            train_fold(df, config, device, fold, audio_image_store, logger)



###MAIN LOOP
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BirdClef2021')
    parser.add_argument("--train-one-fold", action='store_true',
            help="Train one fold")
    parser.add_argument("--model-name", type=str, required=True,
            help="model name")
    args = parser.parse_args()

    #Overwrite config with user-defined args
    config = GlobalConfig
    config.MODEL_NAME = args.model_name

    #initialise logger
    logger = log(config, 'training')
    logger.info(config.__dict__)
    logger.info(args)

    seed_everything(config.SEED)
    df = pd.read_csv(config.CSV_PATH, nrows=None)
    df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)
    LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}

    #load image
    if config.LOAD_FROM_MEM:
        logger.info("Loading spectrogram images to memory...")
        audio_image_store = load_data(df, config)
    else:
        audio_image_store = None

    logger.info("-------------------------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(df, config, device, audio_image_store, logger, train_one_fold=args.train_one_fold)
