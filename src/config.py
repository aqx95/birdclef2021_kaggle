from pathlib import Path

class GlobalConfig:
    SEED = 2020
    BATCH_SIZE = 100
    NUM_WORKERS = 2
    NUM_CLASSES = 397
    SR = 32000
    DURATION = 7
    MAX_READ_SAMPLES = 10
    NUM_EPOCHS = 20
    VERBOSE = True
    LOAD_FROM_MEM = False
    NUM_FOLDS = 5
    MIXUP = False
    SPEC_AUG = True

    TRAIN_STEP_SCHEDULER = False
    VAL_STEP_SCHEDULER = True

    TRAIN_IMAGE_PATH = '/content/audio_images'
    CSV_PATH = '/content/rich_train_metadata.csv'
    SAVE_PATH = '../../save'
    LOG_PATH = '../../logs'

    MODEL_NAME = 'resnet50'
    PRETRAINED = True
    DROP_RATE = 0.0
