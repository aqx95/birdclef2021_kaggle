from pathlib import Path


class GlobalConfig:
    SEED = 2020
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    NUM_CLASSES = 397 #add nocall + 1
    SR = 32000
    DURATION = 7
    MAX_READ_SAMPLES = 10
    NUM_EPOCHS = 20
    VERBOSE = True
    LOAD_FROM_MEM = False
    NUM_FOLDS = 5
    SECONDARY_ID = False
    RESIZE = False
    USE_NOCALL = False
    FIRST_LAST = False

    WARMUP_PROB = 0.05
    MIXUP = False
    SPEC_AUG = False
    AUGMENT = False

    USE_WEIGHT = False
    TRAIN_STEP_SCHEDULER = True
    VAL_STEP_SCHEDULER = False

    TRAIN_COLS = ['primary_label', 'secondary_labels', 'label_id', 'filename', 'fold']
    NOCALL_COLS = ['primary_label', 'secondary_labels', 'label_id', 'itemid', 'fold']

    TRAIN_IMAGE_PATH = '/content/audio_images'
    NOCALL_IMAGE_PATH = '/content/soundscape_nocall_images'
    TRAIN_CSV_PATH = '/content/drive/Shareddrives/Deep Learning/rich_train_metadata.csv'
    NOCALL_CSV_PATH = '/content/nocall_train_metadata.csv'
    SAVE_PATH = '../../save'
    LOG_PATH = '../../logs'

    MODEL_NAME = 'resnet50'
    PRETRAINED = True
    DROP_RATE = 0.0
