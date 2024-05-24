package backprop

func  update(input []float64, target, learningRate float64) {
	// Calculate the output of the neuron
	output := n.forward(input)

	z := sigmoidPrime(output)

	// Calculate the error
	// error := math.pow(target - output , 2)
	// gradient := error - sigmoidPrime(output)

	// Calculate the derivative of the loss with respect to prediction (dLoss/dYPred)
	dLossDYPred := 2.0 * (z - target)

	// Initialize the gradient slice for weights
	//gradW := make([]float64, len(W))

	// Calculate the gradient for each weight and update the weights
	for i := range n.weights {
		// Partial derivative of loss with respect to weight (dLoss/dW)
		//gradW[i] = dLossDYPred * input[i]

		// Update the weight
		n.weights[i] += learningRate * dLossDYPred * input[i]
		// n.weights[i] += learningRate * gradient * input[i]
	}

	// Calculate the gradient for the bias and update the bias
	// Partial derivative of loss with respect to bias (dLoss/dB) is the same as dLossDYPred
	b += learningRate * dLossDYPred
}
