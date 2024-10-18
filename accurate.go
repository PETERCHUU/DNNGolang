package Golang_NN

func (c Chain) calculateAccurate(inputs, targets [][]float64) float64 {
	var accurate float64
	if len(inputs) != len(targets) {
		panic("dataFormate error, input len != target len")
	}
	for i := range inputs {
		predict := c.Predict(inputs[i])

		accurate += Accurate(predict, targets[i])
	}
	accurate /= float64(len(inputs))
	return accurate
}
