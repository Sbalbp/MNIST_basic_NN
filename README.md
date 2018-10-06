
# MNIST_basic_NN
[![CircleCI](https://circleci.com/gh/Sbalbp/MNIST_basic_NN/tree/master.svg?style=svg)](https://circleci.com/gh/Sbalbp/MNIST_basic_NN/tree/master)

Basic implementation of a Neural Network for training on the popular MNIST dataset.

Two possible approaches are available:
1. __[Simple Network](https://en.wikipedia.org/wiki/Artificial_neural_network):__
A simple 3-layered neural network trained over the full training dataset using backpropagation and gradient descent.

2. __[Bagging ensemble](https://en.wikipedia.org/wiki/Bootstrap_aggregating):__
Several 3-leayered neural networks trained each over different random subsets (60% of the size) of the training dataset and aggregating their results to produce the final prediction in order to reduce the overfitting incurred by the original model.

## Usage

First of all, download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and unzip it into a local directory.

After this, you can just run the script:
```
python mnist_nn.py --nn
```

The following arguments are accepted by the script:
* __--help__ Display help.
* __--nn__ Indicates that training with a simple 3-layer Neural Network should be performed.
* __--bag__ Indicates that training using bagging with several models should be performed.
* __-d *dir*__ Indicates the directory where the dataset .idx files are located. By default it assumes the local directory.
* __-lr *lr*__ Indicates what the learning rate of the neural network should be. Default value is 0.2.
* __-n *n*__ Indicates how many different model should be built when using bagging. Default value is 15.
* __-e *e*__ Indicates for how many epochs each neural network should be trained (divided by 10 for bagging). Default value is 1000.

## License

MIT License


