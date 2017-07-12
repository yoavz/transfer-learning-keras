import caltech_101

import argparse
import datetime
import h5py
import keras.applications
import numpy as np
import os

CODES_BATCH_SIZE = 64

def batch_loop(model, dataset, batch_size):
    num_examples = dataset.shape[0]
    num_batches = num_examples / batch_size + 1
    results = []
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b+1) * batch_size, num_examples)
        print "Generating batch {}/{}".format(b, num_batches)
        results.append(model.predict(dataset[start_idx:end_idx, :, :, :]))
    return np.concatenate(results)

def codes_path(model_name, data_dir):
    if model_name == 'vgg19':
        return os.path.join(data_dir, "vgg19_codes.hdf5")
    elif model_name == 'vgg16':
        return os.path.join(data_dir, "vgg16_codes.hdf5")
    elif model_name == 'inception':
        return os.path.join(data_dir, "inception_codes.hdf5")

def generate_codes(model_name, data_dir, batch_size):
    (train_x, train_y), (test_x, test_y) = caltech_101.load_data()
    img_shape = train_x.shape[1:]

    cnn_code_file = codes_path(model_name, data_dir)

    if model_name == 'vgg19':
        model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False,
                input_shape=img_shape)
        preprocess_input = keras.applications.vgg19.preprocess_input
    elif model_name == 'vgg16':
        model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
                input_shape=img_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif model_name == 'inception':
        model = keras.applications.inception_v3.InceptionV3(weights='imagenet',
                include_top=False, input_shape=img_shape)
        preprocess_input = keras.applications.inception_v3.preprocess_input

    # Generate the codes
    print "Generating training codes..."
    train_x_codes = batch_loop(model, preprocess_input(train_x), batch_size)
    print "Generating test codes..."
    test_x_codes = batch_loop(model, preprocess_input(test_x), batch_size)

    f = h5py.File(cnn_code_file, mode="w")
    f.create_dataset("train_x", data=train_x_codes)
    f.create_dataset("train_y", data=train_y)
    f.create_dataset("test_x", data=test_x_codes)
    f.create_dataset("test_y", data=test_y)
    f.close()

def get_codes(model_name, data_dir, batch_size):
    path = codes_path(model_name, data_dir)
    if not os.path.exists(path):
      generate_codes(model_name, data_dir, batch_size)

    f = h5py.File(path, mode="r")
    train_x = f["train_x"][:]
    test_x = f["test_x"][:]
    train_y = f["train_y"][:]
    test_y = f["test_y"][:]
    f.close()
    return ((train_x, train_y), (test_x, test_y))

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_name", type=str, default="vgg19",
                    choices=["vgg16", "vgg19", "inception"])
parser.add_argument("--batch_size", type=int, default=32)

if __name__ == "__main__":
    args = parser.parse_args()
    (train_x, train_y), (test_x, test_y) = get_codes(args.model_name, "data",
                                                     args.batch_size)
