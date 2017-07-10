from keras.preprocessing import image

import h5py
import json
import numpy as np
import os
import pickle

TRAIN_TEST_RATIO = 0.7
CALTECH_101_CACHE = "data/caltech_101.hdf5"
CALTECH_101_DIR = "data/caltech_101"
LABELS = os.path.join(CALTECH_101_DIR, "labels.json")

def get_directories():
    return [ x[0] for x in os.walk(CALTECH_101_DIR) ][1:]

def class_labels():
    if not os.path.isfile(LABELS):
        all_directories = get_directories()
        labels = {idx: os.path.basename(d) for idx, d in enumerate(all_directories)}
        with open(LABELS, "w") as f:
            json.dump(labels, f)

    return json.load(open(LABELS, "r"))

def num_class_labels():
    return len(class_labels())

def load_images():
    all_directories = get_directories()
    labels = class_labels()
    num_labels = len(labels)

    train_x_arr = [] 
    train_y_arr = []
    test_x_arr = [] 
    test_y_arr = []

    for class_idx, name  in labels.iteritems():   
        print "Loading {} images...".format(name)
        class_path = os.path.join(CALTECH_101_DIR, name)
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                    if os.path.isfile(os.path.join(class_path, f))]
        imgs = []
        for img_file in files:
            img = image.load_img(img_file, target_size=(224, 224))
            imgs.append(image.img_to_array(img))

        split = int(np.floor(len(imgs) * TRAIN_TEST_RATIO))

        # each X will be a batch of images
        train_x_arr.append(np.stack(imgs[:split], axis=0))
        test_x_arr.append(np.stack(imgs[split:], axis=0))

        # Generate one-hot encodings for the class labels
        train_batch_len = len(imgs[:split])
        train_y = np.zeros((train_batch_len, num_labels), dtype=int)
        train_y[:, int(class_idx)] = 1
        train_y_arr.append(train_y)

        test_batch_len = len(imgs[split:])
        test_y = np.zeros((test_batch_len, num_labels), dtype=int)
        test_y[:, int(class_idx)] = 1
        test_y_arr.append(test_y)

    return ((np.concatenate(train_x_arr), np.concatenate(train_y_arr)),
            (np.concatenate(test_x_arr), np.concatenate(test_y_arr)))

def load_data(recache=False):
    if recache or not os.path.isfile(CALTECH_101_CACHE):
        (train_x, train_y), (test_x, test_y) = load_images()
        print "Saving file..."
        f = h5py.File(CALTECH_101_CACHE, mode="w")
        f.create_dataset("train_x", data=train_x)
        f.create_dataset("train_y", data=train_y)
        f.create_dataset("test_x", data=test_x)
        f.create_dataset("test_y", data=test_y)
        f.close()

    f = h5py.File(CALTECH_101_CACHE, mode="r")
    train_x = f["train_x"][:]
    test_x = f["test_x"][:]
    train_y = f["train_y"][:]
    test_y = f["test_y"][:]
    f.close()
    return ((train_x, train_y), (test_x, test_y))
