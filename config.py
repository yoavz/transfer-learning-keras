"""File to store common flags between scripts.
"""

import argparse

parser = argparse.ArgumentParser(
    description="Transfer learning experiments with keras.")

# Model configuration
parser.add_argument("--models_dir", type=str, default="models",
                    help="Directory to store model checkpoints")
parser.add_argument("--model_name", type=str, default="vgg19",
                    choices=["vgg16", "vgg19", "inception"],
                    help="The name of the CNN model to use to generate cnn codes")
parser.add_argument("--optimizer", type=str, default="adadelta",
                    choices=["adadelta", "rmsprop"],
                    help="The type of optimizer to use when training")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=50)

# Dataset configuration
parser.add_argument("--data_dir", type=str, default="data",
                    help="Directory to store training/test data and cnn codes")
parser.add_argument("--train_test_ratio", type=float, default=0.7,
                    help="The training / test ratio to split the dataset into")
parser.add_argument("--caltech_101_dir", type=str, default="caltech_101",
                    help="The subdirectory of the caltech_101 data ")
parser.add_argument("--caltech_101_cache", type=str, default="caltech_101.hdf5",
                    help="The filename to store the compiled training/test data")
parser.add_argument("--labels_file", type=str, default="labels.json",
                    help="The filename to store the class indices to names")

args = parser.parse_args()
