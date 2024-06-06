package nnfcgolang

import "errors"

// target function
/*
	output,err:=network.Predict(data)
*/
// one data at a time
func (c Chain) Predict(data []float32) ([]float32, error) {
	if len(data) != len(*(*c.Layers)[0].Neurons) {
		return nil, errors.New("data length not match")
	}
	for i, _ := range *c.Layers {
		data = c.predict(data, i)
	}
	return data, nil
}

func (c Chain) predict(data []float32, index int) []float32 {

	for _, n := range *(*c.Layers)[index].Neurons {
		i := 0
		thisData := data[i]
		data[i] = (*n.Weights)[i] * thisData
		i++
		for i < len(*(*c.Layers)[index].Bias) {
			data[i] *= (*n.Weights)[i] * thisData
			i++
		}
	}
	for i, b := range *(*c.Layers)[index].Bias {
		data[i] += b
	}
	data = (*c.Layers)[index].Activation(data[:len(*(*c.Layers)[index].Bias)])
	//print(len(data), "\n")

	// last output function

	return data
}
