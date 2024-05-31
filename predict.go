package nnfcgolang

// target function
/*
	output,err:=network.Predict(data)
*/
// one data at a time
func (c Chain) Predict(data []float32) []float32 {
	for i, _ := range *c.Layers {
		data = c.predict(data, i)
	}
	return data
}

func (c Chain) predict(data []float32, index int) []float32 {

	for _, n := range *(*c.Layers)[index].Neurons {
		i := 0
		thisData := data[i]
		data[i] = (*n.Weights)[i] * thisData
		i++
		for i < (*c.Layers)[index].NextLen {
			data[i] *= (*n.Weights)[i] * thisData
			i++
		}
	}
	for i, b := range *(*c.Layers)[index].Bias {
		data[i] += b
	}
	data = (*c.Layers)[index].Activation(data[:(*c.Layers)[index].NextLen])
	print(len(data), "\n")

	return data
}
