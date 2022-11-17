import matplotlib.pyplot as plt
import config
import tensorflow as tf

def visualize_training_progress(model_history, epochs):

  train_acc = model_history['accuracy']
  val_acc = model_history['val_accuracy']
  train_loss = model_history['loss']
  val_loss = model_history['val_loss']

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 2, 1)
  plt.plot(range(epochs), train_acc, label='Training Accuracy')
  plt.plot(range(epochs), val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(range(epochs), train_loss, label='Training Loss')
  plt.plot(range(epochs), val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()


def get_picture_title(true_label, model_probs):
  text = ""
  for i in range(len(model_probs)):
    hyper_info = "".join([config.CLASS_NAMES[model_probs[i][0]], ': ', f'{model_probs[i][1]:.2f}%'])
    text = "".join([text, f'{hyper_info:20}\n'])
  text = "".join([text, f'actual: {config.CLASS_NAMES[true_label]}'])
  return text


def visualize_evaluation(model_probs, images, true_labels):

  plt.figure(figsize=(20, 52))
  for i, (image, label) in enumerate(zip(images, true_labels)):
    ax = plt.subplot(8, 4, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title(get_picture_title(label, model_probs[i]))
    plt.axis("off")


def get_top_predictions(num, predictions):
  map_probs = lambda probs_array: [(i, prob) for i, prob in enumerate(probs_array)] 
  prediction_probs = tf.nn.softmax(predictions)
  probs_indices = [map_probs(probs_array) for probs_array in prediction_probs.numpy()]
  return [sorted(x, key=lambda tup: tup[1], reverse=True)[:num] for x in probs_indices]
