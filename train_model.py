from keras.applications import vgg19
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import to_categorical

import argparse
import caltech_101
import datetime
import generate_cnn_codes
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--models_dir", type=str, default="models")
parser.add_argument("--model_name", type=str, default="vgg19", choices=["vgg16", "vgg19", "inception"])
parser.add_argument("--optimizer", type=str, default="adadelta", choices=["adadelta", "rmsprop"])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=200)

if __name__ == "__main__":
  args = parser.parse_args()

  (train_x, train_y), (test_x, test_y) = generate_cnn_codes.get_codes(
      args.model_name, args.data_dir, args.batch_size)

  train_labels = to_categorical(train_y, num_classes=caltech_101.num_class_labels())
  test_labels = to_categorical(test_y, num_classes=caltech_101.num_class_labels())

  input_shape = train_x.shape[1:]
  assert train_x.shape[1:] == test_x.shape[1:]


  inputs = Input(input_shape)
  flatten = Flatten(name = "flatten")(inputs)
  predictions = Dense(caltech_101.num_class_labels(),
                      activation = "softmax")(flatten)

  print "Training model: {}, input shape: {}, flattened: {}".format(
      args.model_name, predictions)

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
            verbose = 1)
  print("Training time: {}".format(datetime.datetime.now() - t))

  loss, accuracy = model.evaluate(test_x, test_y, batch_size=args.batch_size)
  print('Test loss:', loss)
  print('Test accuracy:', accuracy)
