# Neural-Network-in-Python
Multilayer Perceptron implemented in python.

A project I worked on after creating the MNIST_NeuralNetwork project. While C++ was familiar and thus a great way to delve into Neural Networks, it is clear that numpy's ability to quickly perform matrix operations provides Python a clear advantage in terms of both speed and ease when implementing Neural Networks. This project is meant to be a simple Multilayer Perceptron with the added benefit of being able to test multiple different activation functions on a given dataset. The Neural Network also allows for the output layer to be activated using a softmax function, allowing for testing in classification problems as well. The Neural Network is built as a class so that various architectures can also be explored - allowing for variable numbers of hidden layers and hidden nodes. I do regret not having the time to add more flexibility to the network - such as adding the ability to create multiple hidden layers, each with a different number of hidden layer nodes and adding optimizers such as ADAM or RMSProp - however this stands as a good first attempt.

I drew every equation from the source below, certainly one of the best explanations of backpropogation which I encountered.
http://neuralnetworksanddeeplearning.com/chap2.html
