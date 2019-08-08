import numpy as np

class NeuralNetwork:
	def __init__(self, input_layer_size, num_hidden_nodes, num_hidden_layers, output_size, lr, activation_function="sigmoid", output_activation=""):
		self.alpha = lr
		self.activation_function = activation_function
		#i-h-h-h-o Wn = Hn+1
		self.weight_layers = []
		self.weight_layers.append(np.random.rand(num_hidden_nodes, input_layer_size))

		for i in range(num_hidden_layers-1):
			self.weight_layers.append(np.random.rand(num_hidden_nodes,num_hidden_nodes))

		self.weight_layers.append(np.random.rand(output_size,num_hidden_nodes))

		#apply xavier normalization if we're using sigmoid function
		if self.activation_function == "sigmoid":
			for layer in range(len(self.weight_layers)):
				for n2 in range(len(self.weight_layers[layer])):
					fan_in = len(self.weight_layers[layer][n2])
					for n1 in range(len(self.weight_layers[layer][n2])):
						self.weight_layers[layer][n2][n1] *= np.sqrt(1.0/fan_in)

		self.output_activation = output_activation



		#bias nodes
		self.biases = []
		for hl in range(num_hidden_layers+1):
			self.biases.append(np.zeros(num_hidden_nodes))

		self.biases.append(np.zeros(output_size))

	def activation(self, x, out_act=False):
		if out_act and self.output_activation == "softmax":
			osum = np.exp(x).sum()
			return np.exp(x) / osum

		if self.activation_function == "sigmoid":
			exp = np.exp(x)
			return exp / (exp + 1)
		elif self.activation_function == "relu":
			res = x
			for i in range(len(res)):
				res[i] = max(0,res[i])
			return res
		elif self.activation_function =="leakyrelu":
			res = x
			for i in range(len(res)):
				res[i] = max(0.01*res[i], res[i])
			return res
		elif self.activation_function == "tanh":
			return np.tanh(x)

	def activation_derivative(self, x, out_act=False):
		if out_act and self.output_activation == "softmax":
			res = self.activation(x, True)
			return res * (1.0 - res)

		if self.activation_function == "sigmoid":
			sig = self.activation(x)
			return sig * (1 - sig)
		elif self.activation_function == "relu":
			res = x
			for i in range(len(res)):
				if res[i] > 0:
					res[i] = 1
				else:
					res[i] = 0
			return res
		elif self.activation_function == "leakyrelu":
			res = x
			for i in range(len(res)):
				if res[i] > 0:
					res[i] = 1
				else:
					res[i] = 0.01
			return res
		elif self.activation_function == "tanh":
			return 1 - np.square(np.tanh(x))
		

	def feed_forward(self, input):
		#input through hidden layers
		self.input_layer = input
		self.hidden_layers = []
		self.hidden_layers.append(np.dot(self.weight_layers[0], self.input_layer))
		for hidden_layer in range(1,len(self.weight_layers)-1):
			self.hidden_layers.append(np.dot(self.weight_layers[hidden_layer], self.activation(self.hidden_layers[hidden_layer-1])) + self.biases[hidden_layer])

		#hidden layers through output layer
		self.output_layer = np.dot(self.weight_layers[len(self.weight_layers)-1], self.activation(self.hidden_layers[len(self.hidden_layers)-1])) + self.biases[len(self.biases)-1]

		return self.activation(self.output_layer, True)

	def back_propogate(self, y=[], oe=[]):
		#error from output -> backwards
		# BP-1
		out_error = []
		if len(y) == 0 and len(oe) != 0:
			out_error = oe
		else:
			out_error = np.multiply((self.activation(self.output_layer, True) - y), self.activation_derivative(self.output_layer, True))

		#BP-2
		#error from output to last hidden
		hidden_error = np.zeros([len(self.hidden_layers), len(self.hidden_layers[0])])
		hidden_error[len(hidden_error)-1] = np.multiply(np.dot(self.weight_layers[len(self.weight_layers)-1].transpose(), out_error), self.activation_derivative(self.hidden_layers[len(self.hidden_layers)-1]))

		#error from last hidden to first hidden layer
		for i in range(len(hidden_error)-1):
			hidden_layer = len(hidden_error) - i - 2
			hidden_error[hidden_layer] = np.multiply(np.dot(self.weight_layers[hidden_layer+1].transpose(), hidden_error[hidden_layer+1]), self.activation_derivative(self.hidden_layers[hidden_layer]))

		#adjust weights/biases wrt error
		#in to h1
		self.weight_layers[0] -= self.alpha * np.dot(np.reshape(hidden_error[0], (len(hidden_error[0]), -1)), np.reshape(self.activation(self.input_layer), (len(self.input_layer), -1)).transpose())
		self.biases[0] -= self.alpha * hidden_error[0]

		#h1 to hn
		for layer in range(1, len(self.weight_layers)-1):
			self.weight_layers[layer] -= self.alpha * (np.dot(hidden_error[layer], self.activation(self.hidden_layers[layer-1]).transpose()))
			self.biases[layer] -= self.alpha * hidden_error[layer]

		#hn to out
		self.weight_layers[len(self.weight_layers)-1] -= self.alpha * np.dot(np.reshape(out_error, (len(out_error), -1)), np.reshape(self.activation(self.hidden_layers[len(self.hidden_layers)-1]), (len(self.hidden_layers[len(self.hidden_layers)-1]), -1)).transpose())
		self.biases[len(self.biases)-1] -= self.alpha * out_error
		#adjusts done

	def train(self, train_data, labels, iterations, batch_size=0):
		assert(len(train_data) != len(labels)), "Training data and labels of different dimensions"
		if batch_size == 0:
			loss = np.zeros(iterations)
			for iteration in range(iterations):
				for i in range(len(train_data)):
					self.feed_forward(train_data[i])
					self.back_propogate(labels[i])
		else:
			batch_no = np.ceil(len(train_data) / batch_size)
			train = train_data
			l = labels
			for iteration in range(iterations+1):
				train, l = self.uniform_shuffle(np.array(train), np.array(l))
				train_batches = np.array_split(train, batch_no)
				label_batches = np.array_split(l, batch_no)
				for batch in range(len(train_batches)):
					avg_error = np.zeros(len(labels[0]))
					for batch_item in range(len(train_batches[batch])):
						self.feed_forward(train_batches[batch][batch_item])
						#feed into average error calculcation
						avg_error = avg_error + np.multiply((self.activation(self.output_layer, True) - label_batches[batch][batch_item]), self.activation_derivative(self.output_layer, True))
					avg_error = avg_error / batch_no
					self.back_propogate(oe=avg_error)

	def predict(self, input):
		return self.feed_forward(input)

	#from: https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
	def uniform_shuffle(self, a, b):
		p = np.random.permutation(len(a))
		return a[p], b[p]

	#used for testing while building the net
	def loss_function(self, x, y):
		if self.output_activation == "softmax":
			return np.sum(-1 * y * np.log(x))
		else:
			return np.sum(0.5 * (x - y) ** 2) / len(x)



#input_layer_size, num_hidden_nodes, num_hidden_layers, output_size, lr, activation_function, output_activation
net = NeuralNetwork(2, 2, 1, 1, 0.01, "relu")#, "softmax")

#the net can be taught, for example, the XOR function
test_data = [[1,1],
			 [1,0],
			 [0,1],
			 [0,0]]

test_labels = [[0],[1],[1],[0]]

net.train(test_data, test_labels, 100000)

print("Predictions: ")
for i in range(len(test_data)):
	print(test_data[i])
	print(net.predict(test_data[i]))


			





