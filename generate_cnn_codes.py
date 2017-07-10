import caltech_101

import argparse
import datetime
import h5py
import keras.applications
import numpy as np
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--model', type=str, default='vgg19', choices=['vgg16', 'vgg19'])
parser.add_argument('--batch_size', type=int, default=32)

def generate_codes(dataset): 
    num_examples = dataset.shape[0]
    num_batches = num_examples / args.batch_size + 1
    results = []
    for b in range(num_batches):
        start_idx = b * args.batch_size
        end_idx = min((b+1) * args.batch_size, num_examples)
        print "Generating batch {}/{}".format(b, num_batches)
        results.append(model.predict(dataset[start_idx:end_idx, :, :, :]))
    return np.concatenate(results)

if __name__ == "__main__":
    args = parser.parse_args()

    (train_x, train_y), (test_x, test_y) = caltech_101.load_data()
    img_shape = train_x.shape[1:]

    if args.model == 'vgg19':
        model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, 
                input_shape=img_shape)
        preprocess_input = keras.applications.vgg19.preprocess_input
        cnn_code_file = os.path.join(args.data_dir, "vgg19_codes.hdf5")
    elif args.model == 'vgg16':
        model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, 
                input_shape=img_shape)
        preprocess_input = keras.applications.vgg16.preprocess_input
        cnn_code_file = os.path.join(args.data_dir, "vgg16_codes.hdf5")

    # Generate the codes
    print "Generating training codes..."
    train_x_codes = generate_codes(preprocess_input(train_x))
    print "Generating test codes..."
    test_x_codes = generate_codes(preprocess_input(test_x))

    f = h5py.File(cnn_code_file, mode="w")
    f.create_dataset("train_x", data=train_x_codes)
    f.create_dataset("train_y", data=train_y)
    f.create_dataset("test_x", data=test_x_codes)
    f.create_dataset("test_y", data=test_y)
    f.close()
