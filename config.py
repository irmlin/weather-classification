import os

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 2
DATA_DIR = os.path.join(os.getcwd(), 'weather-dataset')
CLASS_NAMES = os.listdir(DATA_DIR)
NUM_CLASSES = len(CLASS_NAMES)