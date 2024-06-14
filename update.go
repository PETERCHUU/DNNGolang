package nnfcgolang

import (
	"errors"
	"math"
)

// one target at a time
func (c *Chain) BackProp(input, target []float32, learningRate float32) error {
	// get every act in every layer
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		return err
	}
	if len(PredictLayers) != len(*c.Layers)+1 {
		errors.New("dataFormate error, prediction data len != layer number")
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
			(*(*c.Layers)[i+1].Bias)[j] += target[j] * learningRate

			for k, _ := range PredictLayers[i] {
				(*(*(*c.Layers)[i+1].Neurons)[j].Weights)[k] += target[j] * PredictLayers[i][k] * learningRate
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
	return nil
}

func (c *Chain) Train(input, target []float32, learningRate float32) {
	c.BackProp(input, target, learningRate)
}

func Accurate(predict, target []float32) float32 {
	var sum float32
	for i, _ := range predict {
		sum += 1 - Cost(predict[i], target[i])
	}
	return sum / float32(len(predict))
}

func Cost(predict, target float32) float32 {
	return float32(math.Abs(float64(predict - target)))
}
