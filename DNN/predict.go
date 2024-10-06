package dnn

func (d DNN) Predict(input []float64) []float64 {
	if len(input) > len(*d.Neurons) {
		// doing this as First RNN return
		panic("Data length is less then neurons")
	}
	if len(*(*d.Neurons)[0].Weights) > len(input) {
		slice2 := make([]float64, len(*(*d.Neurons)[0].Weights)-len(input))
		input = append(input, slice2...)
	}

	for _, n := range *d.Neurons {
		i := 0

		thisData := input[i]
		input[i] = (*n.Weights)[i] * thisData
		i++

		for i < len(*d.Bias) {
			input[i] += (*n.Weights)[i] * thisData
			i++
		}
	}

	for i, b := range *d.Bias {
		input[i] += b
	}

	// last output function
	return d.Activation(input[:len(*d.Bias)])
}
