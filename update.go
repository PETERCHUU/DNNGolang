package nnfcgolang

import (
	"errors"
	"math"
)

// one target at a time
func (c *Chain) BackProp(input, target []float32, learningRate float32) ([][]float32, [][][]float32, error) {
	// get every act in every layer
	// 784 - 49 - 23 - 10
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		return nil, nil, err
	}
	if len(PredictLayers) != len(*c.Layers) {
		return nil, nil, errors.New("dataFormate error, prediction data len != layer number")
	}

	// change target to be cost from last layer
	for i := range target {
		target[i] = Cost(PredictLayers[len(PredictLayers)-1][i], target[i]) * 2
	}

	// layer loop from last hidden layer
	// 23 - 49 - 784
	for i := len(PredictLayers) - 2; i > 0; i-- {
		if len(target) != len(PredictLayers[i+1]) {
			println("Data len Error")
		}
		for j := range PredictLayers[i+1] {
			PredictLayers[i+1][j] = PredictLayers[i+1][j] * target[j]
		}

		cost := target

		// loop activation
		PredictLayers[i+1] = (*c.Layers)[i+1].Prime(PredictLayers[i+1])
		//fmt.Printf("weight: %.3f , \n", (*(*c.Layers)[i+1].Neurons)[0].Weights)

		for j := range PredictLayers[i+1] {
			//change bias number by delta
			target[j] = target[j] * PredictLayers[i+1][j]
			(*(*c.Layers)[i+1].Bias)[j] += target[j] * learningRate
			for k := range PredictLayers[i] {
				(*(*(*c.Layers)[i+1].Neurons)[k].Weights)[j] += target[j] * PredictLayers[i][k] * learningRate
			}
		}
		//fmt.Printf("weight: %.3f\n", (*(*c.Layers)[1].Neurons)[0].Weights)

		target = make([]float32, len(PredictLayers[i]))

		// next layer a loop
		for j := range PredictLayers[i] {
			target[j] = 0
			for k := range PredictLayers[i+1] {
				target[j] += cost[k] * (*(*(*c.Layers)[i+1].Neurons)[j].Weights)[k] * PredictLayers[i+1][k]
			}
		}

	}
	return nil, nil, nil
}

func (c *Chain) Train(input, target []float32, learningRate float32) {
	c.BackProp(input, target, learningRate)
}

func Accurate(predict, target []float32) float32 {
	var sum float32
	for i := range predict {
		sum += 1 - Cost(predict[i], target[i])
	}
	return sum / float32(len(predict))
}

func Cost(predict, target float32) float32 {
	return float32(math.Abs(float64(predict - target)))
}

func (c *Chain) UpdateMiniBatch(miniBatchInput, miniBatchTarget [][]float32, sampleRate int, LearningRate float32) error {
	if len(miniBatchInput) != len(miniBatchTarget) {
		return errors.New("dataFormate error, input len != target len")
	}
	nablaB := make([][]float32, len(*c.Layers))
	nablaW := make([][][]float32, len(*c.Layers))
	for i := range *c.Layers {
		nablaB[i] = make([]float32, len(*(*c.Layers)[i].Bias))
		nablaW[i] = make([][]float32, len((*(*c.Layers)[i].Neurons)))
		for j := range nablaW[i] {
			nablaW[i][j] = make([]float32, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
		}
	}
	for i := 0; i < len(miniBatchTarget); i++ {
		deltaNablaB, deltaNablaW, err := c.MiniBatchBackProp(miniBatchInput[i], miniBatchTarget[i])
		if err != nil {
			println(err.Error())
		}
		for k := range *c.Layers {
			for j := range nablaB[k] {
				for h := range nablaW[k][j] {
					nablaB[k][j] += deltaNablaB[k][j][h]
				}
			}
			for j := range nablaW[k] {
				for h := range nablaW[k][j] {
					nablaW[k][j][h] += deltaNablaW[k][j][h]
				}
			}
		}

	}

	for i := range *c.Layers {
		for j := range *(*c.Layers)[i].Bias {
			(*(*c.Layers)[i].Bias)[j] += LearningRate / float32(len(miniBatchInput)) * nablaB[i][j]
		}
		for j := range *(*c.Layers)[i].Neurons {
			for k := range *(*(*c.Layers)[i].Neurons)[j].Weights {
				(*(*(*c.Layers)[i].Neurons)[j].Weights)[k] += LearningRate / float32(len(miniBatchInput)) * nablaW[i][j][k]
			}
		}
	}
	return nil
}

func (c *Chain) MiniBatchBackProp(input, target []float32) ([][][]float32, [][][]float32, error) {
	w := make([][][]float32, len(*c.Layers))
	b := make([][][]float32, len(*c.Layers))
	for i := range *c.Layers {
		w[i] = make([][]float32, len(*(*c.Layers)[i].Neurons))
		b[i] = make([][]float32, len(*(*c.Layers)[i].Neurons))
		for j := range *(*c.Layers)[i].Neurons {
			w[i][j] = make([]float32, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
			b[i][j] = make([]float32, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
		}
	}
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		return nil, nil, err
	}
	if len(PredictLayers) != len(*c.Layers) {
		return nil, nil, errors.New("dataFormate error, prediction data len != layer number")
	}

	// change target to be cost from last layer
	for i := range target {
		target[i] = Cost(PredictLayers[len(PredictLayers)-1][i], target[i]) * 2
	}

	// layer loop from last hidden layer
	// 23 - 49 - 784
	for i := len(PredictLayers) - 2; i > 0; i-- {
		if len(target) != len(PredictLayers[i+1]) {
			println("Data len Error")
		}
		for j := range PredictLayers[i+1] {
			PredictLayers[i+1][j] = PredictLayers[i+1][j] * target[j]
		}

		cost := target

		// loop activation
		PredictLayers[i+1] = (*c.Layers)[i+1].Prime(PredictLayers[i+1])
		//fmt.Printf("weight: %.3f , \n", (*(*c.Layers)[i+1].Neurons)[0].Weights)

		for j := range PredictLayers[i+1] {
			//change bias number by delta
			target[j] = target[j] * PredictLayers[i+1][j]

			for k := range PredictLayers[i] {
				w[i+1][k][j] += target[j] * PredictLayers[i][k]
				b[i+1][k][j] += target[j]
			}
		}
		//fmt.Printf("weight: %.3f\n", (*(*c.Layers)[1].Neurons)[0].Weights)

		target = make([]float32, len(PredictLayers[i]))

		// next layer a loop
		for j := range PredictLayers[i] {
			target[j] = 0
			for k := range PredictLayers[i+1] {
				target[j] += cost[k] * w[i+1][j][k] * PredictLayers[i+1][k]
			}
		}
	}
	return b, w, nil
}
