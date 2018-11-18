import numpy as np

class FullConnectedNN:
    """
    @brief: N hidden layer neural network with batch gradient descent.
            Currently supporting only relu, tanh and sigmoid activation functions.
    """
    def __init__(self, number_of_neurons=None, activations_funcs=None, batch_size = 20, epochs=100, learning_rate=0.1):
        """
        :param number_of_neurons: number of neurons in each layer.
        :param activations_funcs: activations of hidden layers.
        :param batch_size: batch size.
        :param epochs: how many times to do gradient descent.
        :param learning_rate: learning rate.
        """
        assert (number_of_neurons is not None)
        assert (activations_funcs is not None)
        assert (len(number_of_neurons) == len(activations_funcs) - 1)
        self.nn_size = number_of_neurons
        self.afuncs = activations_funcs
        self.batch_size = batch_size
        self.epochs = epochs
        self.alpha = learning_rate
        # all network weights stored here. (W is vector of matrices).
        self.W = []

    @staticmethod
    def __activation(z, func_name):
        """
        :param z: dot(X, W_h).
        :param func_name: function name to compute.
        :return: activation_function(z).
        """
        if func_name == 'relu':
            # relu = max(0, z).
            return np.maximum(z, 0)
        elif func_name == 'tanh':
            # tanh = (e^z - e(-z)) / (e^z + e(-z)).
            return np.tanh(z)
        elif func_name == 'sigmoid':
            # sigmoid = 1 / (1 + e^(-z)).
            return 1 / (1 + np.exp(-z))

    @staticmethod
    def __d_activation(func_out, func_name):
        """
        :param func_out: dot(prev_layer, W_current).
        :param func_name: function name to compute.
        :return: derivative of activation function.
        """
        if func_name == 'relu':
            # d(relu) = 1 if relu > 0 else 0.
            func_o_tmp = func_out.copy()
            func_o_tmp[func_out <= 0] = 0
            func_o_tmp[func_out > 0] = 1
            return func_o_tmp
        elif func_name == 'tanh':
            # d(tanh) = 1 - tanh^2.
            return 1 - func_out * func_out
        elif func_name == 'sigmoid':
            # d(sigmoid) = sigmoid * (1 - sigmoid).
            return func_out * (1 - func_out)

    def __init_weights(self, X, y):
        """
        :param X: input.
        :param y: ground truth.
        :return: list of randomly initialized weights.
        """
        # multiplying by sqrt(2.0 / input_shape_size) for vanishing exploding gradients. (Andrew Ng. deep learning course).
        # put bias in weights.
        self.W.append((np.random.randn(X.shape[1], self.nn_size[0]) * np.sqrt(2.0 / X.shape[1])))
        for l in range(1, len(self.nn_size)):
            self.W.append(np.random.randn(self.nn_size[l - 1] + 1, self.nn_size[l]) * np.sqrt(2.0 / (self.nn_size[l - 1] + 1)))
        self.W.append(np.random.randn(self.nn_size[-1] + 1, y.shape[1]) * np.sqrt(2.0 / (self.nn_size[-1] + 1)))

    @staticmethod
    def __loss(a_out, a):
        """
        :param a_out: network output.
        :param a: desired output.
        :return: sum of squared difference divided by training set size.
        """
        return np.sum(np.square(a_out - a)) / a.shape[0]

    @staticmethod
    def __add_bias(X):
        """
        :param X: input
        :return: concatenate ones to input to get bias term.
        """
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def __d_W_l(self, a, desired_out):
        """
        :param X: input
        :param a: each layer output.
        :param desired_out: ground truth.
        :return: d(Loss) / d(W[r])        = SUM_p((d(Loss) / d(a[l](p))) * (d(a[l](p)) / d(W[r])).
                 d(a[l](p)) / d(W[r])     = SUM_q((d(a[l](p)) / d(a[l-1](q))) * (d(a[l-1](q)) / d(W[r])).
                 ...
                 d(a[r + 1](v)) / d(W[r]) = SUM_k((d(a[r+1](k)) / d(a[r](k))) * (d(a[r](k)) / d(W[r])).
                 d(a[r + 1](v)) / d(W[r]) = (d(a[r+1](k==t)) / d(a[r](k==t))) * (d(a[r](k==t)) / d(W[r])) --> in one weight.
                 -----------------------------------------------------------------------------------------------------------
                 d(a[r+1](k==t)) / d(a[r](k==t)) = (d(func[r+1](z[r+1]) / d(z[r+1])) * W[r+1].
                 d(a[r](k==t)) / d(W[r])         = (a[r+1] * (d(func[r](z[r]) / d(z[r])).
                 d(a[r + 1](v)) / d(W[r])       <=> d_curr
                 -----------------------------------------------------------------------------------------------------------
                 d(Loss) / d(W[r]) = (d(Loss) / d(a[r+1]))  (d(a[r + 1](v)) / d(W[r]))
        """


    def __forward_backward_prop(self, X, y):
        """
        :param X: input.
        :param y: ground truth.
        :return: updated weights.
        """

        a = [X]
        z0 = np.dot(X, self.W[0])
        a.append(self.__activation(z0, self.afuncs[0]))
        for l in range(1, len(self.W)):
            zi = np.dot(a[l], self.W[l][1:])
            a.append(self.__activation(zi, self.afuncs[l]))

        dW = self.__d_W_l(a, y)
        for l in range(1, len(a)):
            self.W[l - 1] -= (self.alpha * np.dot(a[l - 1], dW[l - 1]) / y.shape[0])

    def fit(self, X, y):
        """
        :param X: input. shape = (number_of_examples, features).
        :param y: output. shape = (number_of_examples, classes).
        """
        X = self.__add_bias(X)
        self.__init_weights(X, y)
        batch_iters = int(X.shape[0] / self.batch_size)
        batch_rem = X.shape[0] - self.batch_size * batch_iters

        for i in range(self.epochs):
            # update weights in every batch.
            for j in range(batch_iters):
                x_batch = X[j * self.batch_size : (j + 1) * self.batch_size, :]
                y_batch = y[j * self.batch_size : (j + 1) * self.batch_size, :]
                self.__forward_backward_prop(x_batch, y_batch)
            x_batch = X[-batch_rem:]
            y_batch = y[-batch_rem:]
            self.__forward_backward_prop(x_batch, y_batch)
            # print loss in end of epoch.
            if i % 100 == 0:
                a_o = self.predict_prob(X)
                print("Loss = " + str(self.__loss(a_o, y)))

    def predict_prob(self, X):
        """
        :param X: input to predict.
        :return: output vector of probabilities.
        """
        z0 = np.dot(X, self.W[0])
        a_o = self.__activation(z0, self.afuncs[0])
        for l in range(1, len(self.W)):
            zi = np.dot(a_o, self.W[l])
            a_o = self.__activation(zi, self.afuncs[l])
        return a_o