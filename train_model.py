from keras.applications import vgg19
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import to_categorical

from config import args
import caltech_101
import datetime
import json
import generate_cnn_codes
import numpy as np
import os

def get_model_filepath(examples_per_class):
  return os.path.join(args.data_dir, args.model_name + "_" +
                      str(examples_per_class) +
                      "_weights.{epoch:02d}-{val_acc:.2f}.hdf5")

def train_with_examples_per_class(examples_per_class):
  (train_x, train_y), (test_x, test_y) = generate_cnn_codes.get_codes(
      args.model_name, args.data_dir, args.batch_size)

  indices = []
  for i in range(caltech_101.num_class_labels()):
    class_indices = np.squeeze(np.argwhere(train_y == i))
    num_in_class = len(class_indices)
    class_indices = class_indices[:min(num_in_class, examples_per_class)]
    indices.append(class_indices)

  all_indices = np.concatenate(indices)
  train_x = train_x[all_indices, :, :, :]
  train_y = train_y[all_indices]

  train_labels = to_categorical(train_y, num_classes=caltech_101.num_class_labels())
  test_labels = to_categorical(test_y, num_classes=caltech_101.num_class_labels())

  input_shape = train_x.shape[1:]
  assert train_x.shape[1:] == test_x.shape[1:]

  inputs = Input(input_shape)
  flatten = Flatten(name = "flatten")(inputs)
  predictions = Dense(caltech_101.num_class_labels(),
                      activation = "softmax")(flatten)

  print "Training model: {}, input shape: {}, flattened: {}".format(
      args.model_name, input_shape, np.prod(input_shape))

  checkpoint = ModelCheckpoint(get_model_filepath(examples_per_class),
                               monitor = "val_acc",
                               save_best_only = True, mode = "max")
  model = Model(inputs, predictions, name = "softmax-classification")
  model.compile(optimizer = args.optimizer,
                loss = "categorical_crossentropy",
                metrics = ["accuracy"])
  t = datetime.datetime.now()
  model.fit(train_x, train_labels,
            batch_size = args.batch_size,
            epochs = args.epochs,
            validation_data = (test_x, test_labels),
            shuffle = True,
            callbacks = [checkpoint],
            verbose = 1)

  training_time = (datetime.datetime.now() - t).seconds
  loss, accuracy = model.evaluate(test_x, test_labels, batch_size=args.batch_size)
  print('Test loss:', loss)
  print('Test accuracy:', accuracy)
  print("Training time: {}".format(training_time))

  return (loss, accuracy, training_time)

if __name__ == "__main__":
  results = {}
  for n in [5, 10, 15, 20, 25, 30]:
    results[n] = train_with_examples_per_class(n)

  with open(os.path.join(args.data_dir, args.model_name + "_results.json"), "w") as f:
    json.dump(results, f)
