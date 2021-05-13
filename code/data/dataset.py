import torch
import numpy as np
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

    @staticmethod
    def normalize(image):
        image = image.astype('float32', copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        impath = self.config.TRAIN_IMAGE_PATH/f"{row.primary_label}/{row.filename}.npy"
        image = np.load(str(impath))[:self.config.MAX_READ_SAMPLES]
        #image = self.spec_store[row.filename]

        image = image[np.random.choice(len(image))]
        image = self.normalize(image)

        label = np.zeros(self.num_classes, dtype=np.float32) + 0.0025
        label[row.label_id] = 0.99

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
