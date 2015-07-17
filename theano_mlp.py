
import numpy as np
from sklearn.cross_validation import train_test_split
import theano.tensor as T
import theano
from theano.tensor import nnet as NN
from sklearn.preprocessing import StandardScaler


class LinearRegressionTheano(object):

	def __init__(self, inputs, n_in, n_out = 1):
		# assuming doing regression for only one output dv
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(value=np.zeros((n_in, n_out), dtype = theano.config.floatX), name = 'W')
		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared(value=np.zeros((n_out,), dtype = theano.config.floatX), name = 'b')
		# self.y_pred = T.flatten(T.dot(inputs, self.W) + self.b)
		# self.y_pred = T.flatten(T.tanh(T.dot(inputs, self.W) + self.b))
		self.y_pred = T.flatten(NN.sigmoid(T.dot(inputs, self.W) + self.b))
		self.params = [self.W, self.b]

	def mean_squared_error(self, y):
		return T.mean((self.y_pred - y) ** 2)
		
	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError()
		if y.dtype.startswith('float'):
			return T.sqrt(T.mean((self.y_pred - y) ** 2))
		else:
			raise NotImplementedError()
		
	def predict_unseen(self):
		return self.y_pred

class HiddenLayer(object):
	def __init__(self, rng, inputs, n_in, n_out, W=None, b=None, activation = T.tanh):
		self.inputs = inputs
		if W is None:
			W_values = np.asarray(rng.uniform(
									low=-np.sqrt(6. / (n_in + n_out)), 
									high=np.sqrt(6. / (n_in + n_out)),
									size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == NN.sigmoid:
				W_values *= 4
			W = theano.shared(value = W_values, name = 'W', borrow = True)
		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value = b_values, name = 'b', borrow = True)
		self.W = W
		self.b = b
		
		lin_output = T.dot(inputs, self.W) + self.b
		self.output = (lin_output if activation is None else activation(lin_output))
		
		self.params = [self.W, self.b]
		
class MLP(object):
	def __init__(self, rng, inputs, n_in, n_hidden, n_out):
		self.hiddenLayer = HiddenLayer(rng=rng, inputs=inputs,
										n_in=n_in, n_out=n_hidden,
										activation = T.tanh)
		self.linearRegressionLayer = LinearRegressionTheano(inputs = self.hiddenLayer.output,
															n_in=n_hidden, n_out=n_out)
		self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.linearRegressionLayer.W).sum()
		
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.linearRegressionLayer.W ** 2).sum()
		
		self.mean_squared_error = self.linearRegressionLayer.mean_squared_error
		self.errors = self.linearRegressionLayer.errors
		
		self.params = self.hiddenLayer.params + self.linearRegressionLayer.params

def convert_data_for_training_to_predict_unseen(X, y, borrow = False):
	# X is a two dimensional array
	# y is a one dimensional array
	SEED = 7
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state = SEED)
	
	stan = StandardScaler()
	X_train = stan.fit_transform(X_train)
	X_valid = stan.transform(X_valid)
	
	train_set_x = theano.shared(np.array(X_train, dtype=theano.config.floatX), borrow=borrow)
	train_set_y = theano.shared(np.array(y_train, dtype=theano.config.floatX), borrow=borrow)
	
	valid_set_x = theano.shared(np.array(X_valid, dtype=theano.config.floatX), borrow=borrow)
	valid_set_y = theano.shared(np.array(y_valid, dtype=theano.config.floatX), borrow=borrow)
	
	return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), stan]


def train_and_get_model_props(X, y, cut, n_hid = 1, n_iter = 5000, L2_reg = 0.0001, L1_reg = 0.0):
	## send in the untransformed input features

	datasets = convert_data_for_training_to_predict_unseen(X, y)
	batch_size, learning_rate = 10, 0.01

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	stan = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')  # the features as matrix
	y = T.dvector('y')  # the targets are represented as 1D float vectors
	
	rng = np.random.RandomState(1234)
	
	model = MLP(rng = rng, inputs = x, n_in = X.shape[1], n_hidden=n_hid,n_out = 1) # hard coding number of outs to 1
	
	cost = model.mean_squared_error(y) + L1_reg * model.L1 + L2_reg * model.L2_sqr

	validate_model = theano.function(inputs = [index], outputs = model.errors(y), 
									givens = {
											x: valid_set_x[index * batch_size: (index + 1) * batch_size],
											y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

	gparams = []
	for param in model.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)

	updates = []

	for param, gparam in zip(model.params, gparams):
		updates.append((param, param - learning_rate * gparam))

	train_model = theano.function(inputs = [index], outputs = cost, updates = updates,
									givens = {
											x: train_set_x[index * batch_size: (index + 1) * batch_size],
											y: train_set_y[index * batch_size: (index + 1) * batch_size]})

	
	best_params = []
	answers = []
	for j in range(n_iter):
		for i in xrange(n_train_batches):
			minibatch_avg_cost = train_model(i)
		validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
		answers.append(np.mean(validation_losses))
		if len(answers) > 1:
			if int(answers[-1] * cut) >= int(answers[-2] * cut):
				break
			else:
				best_params = [x.get_value() for x in model.params]
		
	return best_params + [stan]