import sys
sys.path.append("../../cnn2d/")

import joblib
import logging
import argparse
import os, random, gc
import re, time, json
import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval
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
from model.cnn2d import BirdClefCNN
from data.dataset import prepare_loader, BirdClefDataset
from engine import Fitter



#Load spectrogram images to memory
def load_data(df, config):
    def load_row(row):
        if row.label_id == 397:
            impath = os.path.join(config.NOCALL_IMAGE_PATH, f"{row.primary_label}/{row.filename}.npy")
            return row.filename, np.load(str(impath))
        else:
            impath = os.path.join(config.TRAIN_IMAGE_PATH, f"{row.primary_label}/{row.filename}.npy")
            return row.filename, np.load(str(impath))[:config.MAX_READ_SAMPLES]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res

#mapper for secondary labels
def map_id(sec_label, dict):
  for idx, i in enumerate(sec_label):
    if i == 'rocpig1':
      sec_label[idx] = 'rocpig'
  return [dict[label] for label in sec_label]

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

    plt.savefig(os.path.join(config.SAVE_PATH, f'{config.MODEL_NAME}_history_fold_{fold}.png'))


 #Train single fold
def train_fold(df, config, device, fold, audio_image_store, logger):
    model = BirdClefCNN(config.MODEL_NAME, config.NUM_CLASSES, pretrained=config.PRETRAINED).to(device)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    #class weights initialise
    class_weight = torch.zeros((1,config.NUM_CLASSES))
    for i, count in train_df['label_id'].value_counts().sort_index().items():
        class_weight[:,i] = len(train_df)/(count*config.NUM_CLASSES)
    if config.USE_NOCALL:
        class_weight[:, -1] = 1

    #log fold statistics
    logger.info("Fold {}: Number of unique labels in train: {}".format(fold, train_df['primary_label'].nunique()))

    train_data = BirdClefDataset(audio_image_store, train_df, num_classes=config.NUM_CLASSES,
                              sr=config.SR, duration=config.DURATION, config=config, is_train=True)
    valid_data = BirdClefDataset(audio_image_store, valid_df, num_classes=config.NUM_CLASSES,
                              sr=config.SR, duration=config.DURATION, config=config, is_train=False)
    train_loader, valid_loader = prepare_loader(train_data, valid_data, config)

    img, label = train_data[0]
    logger.info("Image size: {}".format(img.shape))

    fitter = Fitter(model, device, config, class_weight)
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
    #Read train
    df = pd.read_csv(config.TRAIN_CSV_PATH, nrows=None)
    df["secondary_labels"] = df["secondary_labels"].apply(literal_eval)
    LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}
    df = df[config.TRAIN_COLS]

    #Read nocall
    if config.USE_NOCALL:
        nocall_df = pd.read_csv(config.NOCALL_CSV_PATH)
        nocall_df["secondary_labels"] = nocall_df["secondary_labels"].apply(literal_eval)
        #nocall_df = nocall_df[config.NOCALL_COLS]
        #nocall_df.columns = config.TRAIN_COLS
        nocall_df = nocall_df[config.TRAIN_COLS]
        #nocall_df = nocall_df.groupby('fold').apply(lambda x: x.sample(n=500)).reset_index(drop = True)

        df = pd.concat([df, nocall_df], axis=0)

    df['secondary_id'] = df['secondary_labels'].apply(lambda x:map_id(x, LABEL_IDS))

    #load image
    if config.LOAD_FROM_MEM:
        logger.info("Loading spectrogram images to memory...")
        audio_image_store = load_data(df, config)
    else:
        audio_image_store = None

    logger.info("-------------------------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(df, config, device, audio_image_store, logger, train_one_fold=args.train_one_fold)
