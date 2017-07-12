"""Utility to manage the loading and labeling of the raw caltech 101 dataset.
"""
from keras.preprocessing import image

from config import args
import h5py
import json
import numpy as np
import os
import pickle

""" Several convienience functions to assist in the loading / configuration """
def caltech_101_cache():
    return os.path.join(args.data_dir, args.caltech_101_cache)
def caltech_101_dir():
    return os.path.join(args.data_dir, args.caltech_101_dir)
def labels_cache():
    return os.path.join(args.data_dir, args.labels_file)
def get_directories():
    return [ x[0] for x in os.walk(caltech_101_dir()) ][1:]

def class_labels():
    """ Return a mapping of label indices -> class name in caltech 101 dataset
        Mapping is saved to cache so that it is consistent across calls.
    """
    if not os.path.isfile(labels_cache()):
        all_directories = get_directories()
        labels = {idx: os.path.basename(d) for idx, d in enumerate(all_directories)}
        with open(labels_cache(), "w") as f:
            json.dump(labels, f)

    return json.load(open(labels_cache(), "r"))

def num_class_labels():
    return len(class_labels())

def load_images():
    """ Load all the images from JPG format into numpy arrays

        Returns ((training data, training labels), (test data, test labels)),
          data shape: [N, 224, 224, 3]
          labels shape: [N]
          where N is the number of examples
    """
    all_directories = get_directories()
    labels = class_labels()
    num_labels = len(labels)

    train_x_arr = []
    train_y_arr = []
    test_x_arr = []
    test_y_arr = []

    for class_idx, name  in labels.iteritems():
        print "Loading {} images...".format(name)
        class_path = os.path.join(caltech_101_dir(), name)
        files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f))]
        imgs = []
        for img_file in files:
            img = image.load_img(img_file, target_size=(224, 224))
            imgs.append(image.img_to_array(img))

        split = int(np.floor(len(imgs) * args.train_test_ratio))

        # each X will be a batch of images
        train_x_arr.append(np.stack(imgs[:split], axis=0))
        test_x_arr.append(np.stack(imgs[split:], axis=0))

        # Generate one-hot encodings for the class labels
        train_batch_len = len(imgs[:split])
        train_y = np.multiply(int(class_idx),
                              np.ones((train_batch_len,), dtype=int))
        train_y_arr.append(train_y)

        test_batch_len = len(imgs[split:])
        test_y = np.multiply(int(class_idx),
                             np.ones((test_batch_len,), dtype=int))
        test_y_arr.append(test_y)

    return ((np.concatenate(train_x_arr), np.concatenate(train_y_arr)),
            (np.concatenate(test_x_arr), np.concatenate(test_y_arr)))

def load_data(recache=False):
    """ Returns the caltech_101 data, in the format described above

        If the data doesn't exist or recache == True, then load the data into a
        numpy array and save it to the cached file. Otherwise, load the data
        directly from the cached file.
    """
    if recache or not os.path.isfile(caltech_101_cache()):
        (train_x, train_y), (test_x, test_y) = load_images()
        print "Saving file..."
        f = h5py.File(caltech_101_cache(), mode="w")
        f.create_dataset("train_x", data=train_x)
        f.create_dataset("train_y", data=train_y)
        f.create_dataset("test_x", data=test_x)
        f.create_dataset("test_y", data=test_y)
        f.close()

    f = h5py.File(caltech_101_cache(), mode="r")
    train_x = f["train_x"][:]
    test_x = f["test_x"][:]
    train_y = f["train_y"][:]
    test_y = f["test_y"][:]
    f.close()
    return ((train_x, train_y), (test_x, test_y))

if __name__ == "__main__":
    load_data(recache=True)
