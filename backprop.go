package nnfcgolang

import "math"

func cost(predict []float32, target []float32) []float32 {
	// placeholder
	if len(predict) != len(target) {
		return nil
	}
	var cost []float32
	for i, p := range predict {
		cost[i] = float32(math.Pow(float64(target[i]-p), 2))
	}
	return cost
}

// gradient descent function

// SDG is a function for SDG optimizer
// input is list of data, target is list of target
// len of inputs and targets must be the same
func (c *Chain) Update(inputs, targets [][]float32) {
	// placeholder
	// check len of input and target
	if len(inputs) == 0 || len(targets) == 0 {
		return
	}

	for i, input := range inputs {
		predict, err := c.Predict(input)
		if err != nil && len(predict) != len(targets[i]) && len(predict) != len(*(*c.Layers)[len(*c.Layers)-1].Neurons) {
			return
		}
	}

}

func (c *Chain) SDG(predict, target []float32) {
	var Cost []float32
	// calculate cost
	Cost = cost(predict, target)
	// calculate gradient
	for i := len(*c.Layers) - 1; i >= 0; i-- {
		z := (*c.Layers)[i].Prime(Cost)
		for j, neuron := range *(*c.Layers)[i].Neurons {
			dLossDYPred := 2.0 * (z[j] - target[j])
			for i, weight := range *neuron.Weights {
				// Partial derivative of loss with respect to weight (dLoss/dW)
				//gradW[i] = dLossDYPred * input[i]

				// Update the weight
				weight += (*c.Layers)[i].LearningRate * dLossDYPred * target[j]
				// n.weights[i] += learningRate * gradient * input[i]
			}
			(*(*c.Layers)[i].Bias)[j] += (*c.Layers)[i].LearningRate * dLossDYPred
		}
	}
}
