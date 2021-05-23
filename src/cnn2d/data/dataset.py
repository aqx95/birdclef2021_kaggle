import os
import torch
import torchvision
import numpy as np
from audiomentations import *

from data.transforms import spec_augment
from torch.utils.data import Dataset, DataLoader

class BirdClefDataset(Dataset):
    def __init__(self, spec_store, df, num_classes, duration, config, sr=32000, is_train=True):
        self.spec_store = spec_store
        self.df = df
        self.sr = sr
        self.config = config
        self.num_classes = num_classes
        self.is_train = is_train
        self.duration = duration
        self.audio_length = duration * sr
        self.transform = spec_augment() if is_train and config.SPEC_AUG else None

    @staticmethod
    def normalize(image):
        image = image.astype('float32', copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.config.LOAD_FROM_MEM:
            image = self.spec_store[row.filename]
        else:
            if row.label_id == 397:
                impath = os.path.join(self.config.NOCALL_IMAGE_PATH, f"{row.primary_label}/{row.filename}.npy")
                image = np.load(str(impath))
            else:
                impath = os.path.join(self.config.TRAIN_IMAGE_PATH, f"{row.primary_label}/{row.filename}.npy")
                image = np.load(str(impath))[:self.config.MAX_READ_SAMPLES]

        image = image[np.random.choice(len(image))]
        if self.transform and np.random.rand()<0.5:
            image = self.transform(image)
        if self.config.RESIZE:
            img = torch.FloatTensor(img).unsqueeze(0)
            mel_spec = torchvision.transforms.Resize((224, img.size()[-1]))(img).squeeze(0).numpy()
        image = self.normalize(image)

        label = np.zeros(self.num_classes, dtype=np.float32)
        label[row.label_id] = 1
        if self.config.SECONDARY_ID:
            label[row.secondary_id] = 0.2

        return image, label


def prepare_loader(train_data, valid_data, config):
    train_loader = DataLoader(train_data,
                    batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS,
                    shuffle=True,
                    pin_memory=True)

    valid_loader = DataLoader(valid_data,
                    batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS,
                    shuffle=False)

    return train_loader, valid_loader
