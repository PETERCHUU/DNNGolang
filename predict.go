package nnfcgolang

import "errors"

var (
	IRnn bool = false
)

/* predict rnn : return [fc, rnn]

append input with 
 */

func (c Chain) Predict(data []float32) []float32 {
	if len(data) == 0 {
		panic("data length is 0")
	}

	for i := range *c.Layers {
		switch *&(*c.Layers)[i].NNtype {
		case FC:
			data = c.FCpredict(data, i)
		case RNN:
			if i != len(*c.Layers)-1 {
				panic("RNN must be the last layer")
			}
			return c.FCpredict(data, i)
		}
	}

	return data
}

func (c Chain) FCpredict(data []float32, index int) []float32 {
	if len(data) != len(*(*c.Layers)[index].Neurons) {
		panic("data length not match")
	}
	for _, n := range *(*c.Layers)[index].Neurons {
		i := 0
		thisData := data[i]
		data[i] = (*n.Weights)[i] * thisData
		i++
		for i < len(*(*c.Layers)[index].Bias) {
			data[i] += (*n.Weights)[i] * thisData
			i++
		}
	}
	for i, b := range *(*c.Layers)[index].Bias {
		data[i] += b
	}

	// last output function
	return (*c.Layers)[index].Activation(data[:len(*(*c.Layers)[index].Bias)])
}

func (c Chain) PredictLayer(data []float32) ([][]float32, error) {
	if len(data) != len(*(*c.Layers)[0].Neurons) {
		return nil, errors.New("data length not match")
	}

	var PredictData [][]float32
	PredictData = append(PredictData, data)
	for i := 1; i < len(*c.Layers); i++ {
		PredictData = append(PredictData, c.FCpredict(PredictData[i-1], i))
	}

	return PredictData, nil
}
