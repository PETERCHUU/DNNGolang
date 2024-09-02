package DNNGolang

import (
	"errors"
)

var (
	IRnn bool = false
)

/*
	predict rnn : return [fc, rnn]

list of input --> one output

append input with
*/
func (c Chain) RNNPredict(DataList [][]float64) []float64 {

	if len(DataList) == 0 {
		panic("Predict input length cannot be 0")
	}
	for i := 0; i < len(DataList)-1; i++ {
		DataList[i+1] = append(DataList[i+1], c.Predict(DataList[i])...)
	}
	return c.Predict(DataList[len(DataList)-1])

}

func (c Chain) Predict(data []float64) []float64 {
	if len(data) == 0 {
		panic("data length is 0")
	}

	for i := range *c.Layers {
		data = c.FCpredict(data, i)
	}

	return data
}

func (c Chain) FCpredict(data []float64, index int) []float64 {
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

func (c Chain) PredictLayer(data []float64) ([][]float64, error) {
	if len(data) != len(*(*c.Layers)[0].Neurons) {
		return nil, errors.New("data length not match")
	}

	var PredictData [][]float64
	PredictData = append(PredictData, data)
	for i := 0; i < len(*c.Layers); i++ {
		PredictData = append(PredictData, c.FCpredict(PredictData[i], i))
	}

	return PredictData, nil
}
