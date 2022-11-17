import os
import tensorflow as tf

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
IMG_HEIGHT_LARGE = 224
IMG_WIDTH_LARGE = 224
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 2
DATA_DIR = os.path.join(os.getcwd(), 'weather-dataset')
CLASS_NAMES = os.listdir(DATA_DIR)
NUM_CLASSES = len(CLASS_NAMES)

def load_data():

  train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
  )

  val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
  )

  test_size = int(len(val_data) / 2)
  test_data = val_data.take(test_size)
  val_data = val_data.skip(test_size)

  return train_data, val_data, test_data

def load_data_larger_shape():

  train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT_LARGE, IMG_WIDTH_LARGE),
    batch_size=BATCH_SIZE
  )

  val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT_LARGE, IMG_WIDTH_LARGE),
    batch_size=BATCH_SIZE
  )

  test_size = int(len(val_data) / 2)
  test_data = val_data.take(test_size)
  val_data = val_data.skip(test_size)

  return train_data, val_data, test_data
