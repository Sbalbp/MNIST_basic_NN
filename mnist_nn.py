import sys
import struct
import argparse
import numpy as np
from builtins import range

# Reads an idx file and stores it in a numpy array
def read_idx( filename ):
	try:
	    with open(filename, 'rb') as f:
	        # First two bytes are ignored, second byte is the data type and last byte is the number of dimensions
	        zero, data_type, dims = struct.unpack( '>HBB', f.read(4) )
	        # Read <dims> integers (4 bytes each) to get the size of each dimension
	        shape = tuple( struct.unpack( '>I', f.read(4) )[0] for d in range(dims) )
	        # Read the values (a byte each) as string, cast to 8-bit unsigned int and reshape numpy array to the correct dimensions
	        return np.fromstring( f.read(), dtype = np.uint8 ).reshape(shape)
	except Exception as err:
		print( 'Error reading IDX file: \'%s\':\n%s' % ( filename, err ) )
		raise

# Simple Neural Network definition
class SimpleNN(object):

    def __init_layers(self, layers, weight_factor = 1):
        self.weights = []
        self.biases = []

        for i in range( len(layers)-1 ):
            self.weights.append(weight_factor * np.random.normal( 0, 1, (layers[i], layers[i+1] )))
            self.biases.append(weight_factor * np.random.normal( 0, 1, (1, layers[i+1] )))

    def __init__(self, input_size, output_size, mid_layers = [300], weight_factor = 1):
        self.__init_layers([input_size] + mid_layers + [output_size], weight_factor)

    # Forward computation
    def forward(self, ipt):
        x = ipt
        self.inputs = []

        for weight, bias in zip(self.weights, self.biases):
            self.inputs.append(x)

            # Multiply by weights and add bias
            wb = np.matmul(x, weight) + bias
            # Logistic activation
            x = 1 / (1 + np.exp(-wb))

        return x

    # Backpropagation of error
    def backward(self, output, expected):
        self.gradients = []

        # Error gradient on the output layer
        grad = output * (1 - output) * (output - expected)
        # Store the gradients
        self.gradients.insert(0, grad)

        for i in range(len(self.weights)-1, 0, -1):
            # Propagate the error
            grad = self.inputs[i] * (1 - self.inputs[i]) * np.reshape( np.sum(grad * self.weights[i], 1), (1, self.weights[i].shape[0]) )
            # Store the gradients
            self.gradients.insert(0, grad)

    # Update weights and biases
    def update(self, lr = 0.5):
        for weight, bias, grad, inpt in zip(self.weights, self.biases, self.gradients, self.inputs):
            weight -= lr * np.matmul(np.transpose(inpt), grad)
            bias -= lr * grad




# Command line parameters
parser = argparse.ArgumentParser( description = 'Learn from the MNIST dataset' )
parser.add_argument( '-d', dest = 'directory', metavar = 'MNIST location', default = '.', help = 'Location of MNIST IDX files (default: current directory)' )
parser.add_argument( '-lr', dest = 'lr', metavar = 'Learning Rate', type = float, default = 0.2, help = 'Learning Rate during training (default: 0.2)' )
parser.add_argument( '-n', dest = 'num_models', metavar = 'Num. Models', type = int, default = 15, help = 'Number of different models to learn in Bagging (default: 15)' )
parser.add_argument( '-e', dest = 'epoch', metavar = 'Epochs', type = int, default = 1000, help = 'Number of epochs to learn for (default: 1000)' )
parser.add_argument( '--nn', dest = 'use_nn', action = 'store_true', help = 'Use Simple 3 layer Neural Network' )
parser.add_argument( '--bag', dest = 'use_bagging', action = 'store_true', help = 'Use bagging of weaker networks' )

args = parser.parse_args()

mnist_dir = args.directory
learning_rate = args.lr
epochs = args.epoch
epochs_bagging = epochs / 10
n_models = args.num_models
use_nn = args.use_nn
use_bagging = args.use_bagging

# Read the datasets
try:
	train_data = read_idx( '%s/train-images.idx3-ubyte' % mnist_dir )
	train_labels = read_idx( '%s/train-labels.idx1-ubyte' % mnist_dir )
	test_data = read_idx( '%s/t10k-images.idx3-ubyte' % mnist_dir )
	test_labels = read_idx( '%s/t10k-labels.idx1-ubyte' % mnist_dir )
except Exception as err:
	sys.exit()


# Standardize the data for faster convergence during training
all_data = np.vstack( (train_data, test_data) )
all_data = ( all_data - all_data.mean() ) / all_data.std()
train_data = all_data[0:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

# Generate one-hot encodings for the labels
nclasses = 10

train_one_hot = np.zeros((train_labels.shape[0], 10))
test_one_hot = np.zeros((test_labels.shape[0], 10))

train_one_hot[ np.arange(train_labels.shape[0]), train_labels] = 1
test_one_hot[ np.arange(test_labels.shape[0]), test_labels] = 1

# Reshape images as 1D arrays
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2] ))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2] ))

# Split train dataset into train (80%) and validation (20%) datasets
train_perc = 0.8
cutoff = int(train_perc * train_data.shape[0])

perm = np.random.permutation(train_data.shape[0])
val_data = train_data[ perm[cutoff:] ]
train_data = train_data[ perm[0:cutoff] ]

val_one_hot = train_one_hot[ perm[cutoff:] ]
train_one_hot = train_one_hot[ perm[0:cutoff] ]


###########################################################
"""

Simple learning with only one neural network.
In each epoch we run the full forward-backward-update pass
through every instance in the training set.
We can observe overfitting with a high enough number of epochs.

Experimental result:
Network trained over 1000 epochs and
learning rate of 0.2 achieves an accuracy of 0.9315

"""
###########################################################

if( use_nn ):

	# Create Neural Network
	nn = SimpleNN( 784, 10, mid_layers = [30], weight_factor = np.sqrt(2) )

	# Simple training
	print('\n-----Starting Training ( Single NN )-----\n')

	# In each epoch...
	for i in range( epochs ):
	    # ...pass through the training set
	    tr_error = 0
	    for j in range(train_data.shape[0]):
	        ipt_data = train_data[j:j+1]

	        fwd = nn.forward(ipt_data)
	        tr_error += np.sum( ( train_one_hot[j:j+1] - fwd )**2 ) / 10
	        nn.backward(fwd, train_one_hot[j:j+1])
	        nn.update( learning_rate )

	    # Pass through the validation set (only forward pass)
	    val_error = 0
	    for j in range(val_data.shape[0]):
	        fwd = nn.forward(val_data[j:j+1])
	        val_error += np.sum(( val_one_hot[j:j+1] - fwd )**2) / 10

	    print( 'epoch %d, train error: %f, val error: %f' % (i, tr_error / train_data.shape[0], val_error / val_data.shape[0]))

	print('\n-----End of Training ( Single NN )-----\n')

	# Accuracy
	print( 'Single NN - Accuracy on test data: %f' % (np.sum( np.argmax( nn.forward(test_data),1 ) == test_labels ) / float( test_data.shape[0] )) )



###########################################################
"""

Learning using an ensemble approach ( bagging ).
Using a single network we observe overfitting, so we try
to learn several weaker classifiers ( using random subsamples
of the training set ) and aggregate their predictions to
obtain a lower variance model that outperforms the previous
result with a single neural network trained over the whole
dataset.

Experimental result:
Ensemble with 15 models trained over 50 epochs each and
learning rate of 0.2 achieves an accuracy of 0.9504

"""
###########################################################

if( use_bagging ):

	print('\n-----Starting Training ( Bagging )-----\n')

	nets = []
	for n_classifier in range( n_models ):
	    print('\nTraining classifier %d\n' % n_classifier)
	    net = SimpleNN( 784, 10, mid_layers = [30], weight_factor = np.sqrt(2) )

	    # Train each network over 60% of the training instances (weak classifier - reduces correlation between models)
	    instance_perm = np.random.permutation( train_data.shape[0] )[ 0:int( train_data.shape[0] * 0.6 ) ]

	    # In each epoch...
	    for i in range( epochs_bagging ):
	    	# ...pass through the training set
	        for j in instance_perm:
	            ipt_data = train_data[j:j+1]

	            fwd = net.forward(ipt_data)
	            net.backward(fwd, train_one_hot[j:j+1])
	            net.update( learning_rate )
	        print('Classifier %d, Epoch %d' % (n_classifier, i))

	    nets.append(net)

	# Training
	print('\n-----End of Training ( Bagging )-----\n')

	# Generate final prediction by summing all of the prediction
	result_sum = np.zeros((10000,10))
	for n_classifier in range(5):
	    result_sum += nets[n_classifier].forward(test_data)
	# Accuracy
	print( 'Bagging - Accuracy on test data: %f' % (np.sum( np.argmax(result_sum, 1) == test_labels ) / float(test_data.shape[0]) ) )

