package dnn

func (d DNN) Predict(data []float64) []float64 {
	if len(data) > len(*d.Neurons) {
		// doing this as First RNN return
		panic("Data length is less then neurons")
	}
	if len(*(*d.Neurons)[0].Weights) > len(data) {
		slice2 := make([]float64, len(*(*d.Neurons)[0].Weights)-len(data))
		data = append(data, slice2...)
	}

	for _, n := range *d.Neurons {
		i := 0

		thisData := data[i]
		data[i] = (*n.Weights)[i] * thisData
		i++

		for i < len(*d.Bias) {
			data[i] += (*n.Weights)[i] * thisData
			i++
		}
	}

	for i, b := range *d.Bias {
		data[i] += b
	}

	// last output function
	return d.Activation(data[:len(*d.Bias)])
}
