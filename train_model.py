""" The main script used to run experiments. """

from keras.applications import vgg19
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.utils import to_categorical

from config import args
import caltech_101
import datetime
import generate_cnn_codes
import json
import numpy as np
import os

def checkpoint_filepath(examples_per_class):
    return os.path.join(args.data_dir, args.model_name + "_" +
                      str(examples_per_class) +
                      "_weights.{epoch:02d}-{val_acc:.2f}.hdf5")

def train_with_examples_per_class(examples_per_class=None):
    """ Train a softmax model on the CNN codes of a model given the arguments
        specified (see config.py for training/model configuration).

        examples_per_class (optional) will limit the training instances per
        class to a maximum number, if specified.

        Returns (test_loss, test_accuracy, training_time) for trained model.
    """

    (train_x, train_y), (test_x, test_y) = generate_cnn_codes.get_codes()

    if examples_per_class:
        # For each class, cutoff the indices at examples_per_class and use
        # these indices to reassign the training data and labels.
        indices = []
        for i in range(caltech_101.num_class_labels()):
            class_indices = np.squeeze(np.argwhere(train_y == i))
            num_in_class = len(class_indices)
            class_indices = class_indices[:min(num_in_class, examples_per_class)]
            indices.append(class_indices)

        all_indices = np.concatenate(indices)
        train_x = train_x[all_indices, :, :, :]
        train_y = train_y[all_indices]

    # Convert the 1-D labels to one-hot matrices
    train_labels = to_categorical(train_y, num_classes=caltech_101.num_class_labels())
    test_labels = to_categorical(test_y, num_classes=caltech_101.num_class_labels())

    # Sanity check: the training data shape should equal the test data shape
    input_shape = train_x.shape[1:]
    assert train_x.shape[1:] == test_x.shape[1:]

    # The model is extremely simple: flatten the CNN convolutional layer to a 1-D layer and
    # train a single softmax layer on top of it.
    inputs = Input(input_shape)
    flatten = Flatten(name = "flatten")(inputs)
    predictions = Dense(caltech_101.num_class_labels(), activation = "softmax")(flatten)

    print "Training model: {}, input shape: {}, flattened: {}".format(
      args.model_name, input_shape, np.prod(input_shape))

    model = Model(inputs, predictions, name = "softmax-classification")
    model.compile(optimizer = args.optimizer,
                  loss = "categorical_crossentropy",
                  metrics = ["accuracy"])

    # Model checkpoints will be saved every time the validation accuracy reaches a new
    # max value.
    checkpoint = ModelCheckpoint(checkpoint_filepath(examples_per_class),
                                 monitor = "val_acc",
                                 save_best_only = True, mode = "max")

    # The main keras training loop that does most of the computation
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
    if args.exp_mode == "num_examples":
        """ The num_examples experiment, which runs the specified model for varying levels
            of maximum training instances per class and saves the results to an output
            file.
        """
        results = {}
        for n in [5, 10, 15, 20, 25]:
            results[n] = train_with_examples_per_class(n)

        with open(os.path.join(args.data_dir, args.model_name + "_results.json"), "w") as f:
            json.dump(results, f)

    else:
        """ The default experiment trains one model as specified by config flags """
        train_with_examples_per_class()
