package nnfcgolang

import "math"

// cost = 2*(prediction - target) * sigmoidPrime(prediction) * weight * sigmoidPrime(hidden) * weight * sigmoidPrime(input)
//(output_activations-y)
// delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
// sp = sigmoid_prime(z)
// delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
/*
def backprop(self, x, y):
            """Return a tuple "(nabla_b, nabla_w)" representing the
            gradient for the cost function C_x.  "nabla_b" and
            "nabla_w" are layer-by-layer lists of numpy arrays, similar
            to "self.biases" and "self.weights"."""
            nabla_b = [np.zeros(b.shape) for b in self.biases] #make([]float32,len(biases))
            nabla_w = [np.zeros(w.shape) for w in self.weights] #make([]float32,len(weights))
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer

            # The following loop is make each layer of the network
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)


            # backward

			for i in xrange(len(y)):
		               delta = self.cost_derivative(activations[-1][i], y[i]) * sigmoid_prime(zs[-1][i])
		               nabla_b[-1] = delta
		               for j in xrange(len(activations[-2])):
		                   nabla_w[-1][j] = delta * activations[-2][j]

            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.


		   for l in xrange(2, self.num_layers):
		           z = zs[-l]
		           sp = sigmoid_prime(z)
		           delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
		           nabla_b[-l] = delta each weight
		           nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) each activation

            return (nabla_b, nabla_w)
*/

// one target at a time
func (c *Chain) BackProp(input, target []float32, learningRate float32) {
	// get every act in every layer
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		println(err)
	}
	if len(PredictLayers) != len(*c.Layers)+1 {
		println("dataFormate error, prediction data len != layer number")
	}

	// change target to be cost from last layer
	for i, _ := range target {
		target[i] = Cost(PredictLayers[len(PredictLayers)-1][i], target[i]) * 2
	}

	// layer loop from last hidden layer
	for i := len(PredictLayers) - 2; i > 0; i-- {
		if len(target) != len(PredictLayers[i+1]) {
			println("Data len Error")
		}
		for j, _ := range PredictLayers[i+1] {
			PredictLayers[i+1][j] = PredictLayers[i+1][j] * target[j]
		}

		cost := target
		// loop activation
		PredictLayers[i+1] = (*c.Layers)[i+1].Prime(PredictLayers[i+1])

		for j, _ := range PredictLayers[i+1] {

			//change bias number by delta
			target[j] = target[j] * PredictLayers[i+1][j]
			(*(*c.Layers)[i+1].Bias)[j] = target[j] * learningRate

			for k, _ := range PredictLayers[i] {
				(*(*(*c.Layers)[i+1].Neurons)[j].Weights)[k] = target[j] * PredictLayers[i][k] * learningRate
			}
		}

		// next layer a loop
		for j, _ := range PredictLayers[i] {
			target[j] = 0
			for k, _ := range PredictLayers[i+1] {
				target[j] += cost[k] * (*(*(*c.Layers)[i+1].Neurons)[k].Weights)[j] * PredictLayers[i+1][k]
			}
		}

	}
}

func Cost(predict, target float32) float32 {
	return float32(math.Abs(float64(predict - target)))
}
