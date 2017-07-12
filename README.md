# Transfer Learning Experiments with Keras

This repository contains several explorations pertaining to transfer learning
(also sometimes referred to as domain adaptation), using ImageNet as a source
dataset and Caltech-101 as a target dataset

## Quick Start

To run these experiments, follow instructions to install 
[keras](https://github.com/fchollet/keras). I've used the tensorflow backend
for these experiments, but any backend should work. You will also need 
install [h5py](http://www.h5py.org/) to cache and load models.

## Experiments

The performance numbers from the following experiments are taken from
experiments on my Macbook Air, with a 2.2 GHz Intel Core i7 and 8 GB of DDR3
RAM.

#### [Inception V3](https://arxiv.org/pdf/1512.00567.pdf)

The final convolutional layer of the inception v3 model has an output shape of 5 x 5 x 2048, 
resulting in a flattened CNN code of 51,200 dimensions. Since this is a quite large softmax
layer, training on my laptop in a short amount of time was more intensive than VGG19. However,
I was able to reach a performance of **90.79%** test accuracy in just **5 epochs of training (129
seconds)**. Quite remarkable for such a small amount of epochs / training data.

#### [VGG19](https://arxiv.org/pdf/1409.1556.pdf)

The final convolutional layer of VGG19 has an output shape of 7 x 7 x 512, or
25,008 flattened.  This model converged more slowly than inception, but I was
able to train it for more epochs due to the smaller size. I got a performance
of **79.08%** test accuracy after **25 epochs of training (328 seconds)**
