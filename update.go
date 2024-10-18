package Golang_NN

import (
	"errors"
	"math"
)

/*

target function

model.TrainBy(adam)

model.windowSize(int)

model.train(data) [][]float32

model.test(testdata)

*/

// one target at a time
func (c *Chain) BackProp(input, target []float64, learningRate float64) ([][]float64, [][][]float64, error) {
	// get every act in every layer
	// 784 - 49 - 23 - 10
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		return nil, nil, err
	}
	if len(PredictLayers) != len(*c.Layers) {
		return nil, nil, errors.New("dataFormate error, prediction data len != layer number")
	}

	// change target to be next layer cost from last layer
	for i := range target {
		target[i] = Cost(PredictLayers[len(PredictLayers)-1][i], target[i]) * 2
	}

	// layer loop from last hidden layer to first layer ,
	// i+1 == next layer , i== this layer
	// 23 - 49 - 784
	for i := len(PredictLayers) - 2; i > 0; i-- {
		if len(target) != len(PredictLayers[i+1]) {
			println("Data len Error")
		}

		// using next layer cost correcting the next layer prediction??
		for j := range PredictLayers[i+1] {
			PredictLayers[i+1][j] = PredictLayers[i+1][j] * target[j]
		}

		// loop activation
		//PredictLayers[i+1] == exposed layer
		PredictLayers[i+1] = (*c.Layers)[i+1].Prime(PredictLayers[i+1])
		//fmt.Printf("weight: %.3f , \n", (*(*c.Layers)[i+1].Neurons)[0].Weights)

		for j := range PredictLayers[i+1] {

			// using next layer cost to correcting exposed next layer
			PredictLayers[i+1][j] = target[j] * PredictLayers[i+1][j]

			//change next layer bias number by corrected exposed next layer
			(*(*c.Layers)[i+1].Bias)[j] += PredictLayers[i+1][j] * learningRate

			//change next layer weight number by this layer prediction and corrected exposed next layer
			for k := range PredictLayers[i] {
				(*(*(*c.Layers)[i+1].Neurons)[k].Weights)[j] += PredictLayers[i+1][j] * PredictLayers[i][k] * learningRate
			}
		}
		//fmt.Printf("weight: %.3f\n", (*(*c.Layers)[1].Neurons)[0].Weights)

		// count this layer Cost
		Cost := make([]float64, len(PredictLayers[i]))

		// next layer a loop
		for j := range PredictLayers[i] {
			for k := range PredictLayers[i+1] {
				Cost[j] += target[k] * (*(*(*c.Layers)[i+1].Neurons)[j].Weights)[k] * PredictLayers[i+1][k]
			}
		}

		target = Cost

	}
	return nil, nil, nil
}

func Cost(predict, target float64) float64 {
	return (target - predict) * 2
}

func (c *Chain) UpdateMiniBatch(miniBatchInput, miniBatchTarget [][]float64, sampleRate int, LearningRate float64) error {
	if len(miniBatchInput) != len(miniBatchTarget) {
		return errors.New("dataFormate error, input len != target len")
	}
	nablaB := make([][]float64, len(*c.Layers))
	nablaW := make([][][]float64, len(*c.Layers))
	for i := range *c.Layers {
		nablaB[i] = make([]float64, len(*(*c.Layers)[i].Bias))
		nablaW[i] = make([][]float64, len((*(*c.Layers)[i].Neurons)))
		for j := range nablaW[i] {
			nablaW[i][j] = make([]float64, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
		}
	}
	for i := 0; i < len(miniBatchTarget); i++ {
		deltaNablaB, deltaNablaW, err := c.SingleBackProp(miniBatchInput[i], miniBatchTarget[i])
		if err != nil {
			println(err.Error())
		}

		//get mean of delta

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
			(*(*c.Layers)[i].Bias)[j] += (LearningRate / float64(len(miniBatchInput)) * nablaB[i][j])
		}
		for j := range *(*c.Layers)[i].Neurons {
			for k := range *(*(*c.Layers)[i].Neurons)[j].Weights {
				(*(*(*c.Layers)[i].Neurons)[j].Weights)[k] += (LearningRate / float64(len(miniBatchInput)) * nablaW[i][j][k])
			}

		}
	}
	return nil
}

func (c *Chain) SingleBackProp(input, target []float64) ([][][]float64, [][][]float64, error) {

	w := make([][][]float64, len(*c.Layers))
	b := make([][][]float64, len(*c.Layers))
	for i := range *c.Layers {
		w[i] = make([][]float64, len(*(*c.Layers)[i].Neurons))
		b[i] = make([][]float64, len(*(*c.Layers)[i].Neurons))
		for j := range *(*c.Layers)[i].Neurons {
			w[i][j] = make([]float64, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
			b[i][j] = make([]float64, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
		}
	}

	// layer input- hidden - output
	PredictLayers, err := c.PredictLayer(input)
	if err != nil {
		return nil, nil, err
	}
	if len(PredictLayers) != len(*c.Layers)+1 {
		return nil, nil, errors.New("dataFormate error, prediction data len != layer number")
	}

	// layer loop from last hidden layer
	//  23 - 49 - 784
	for i := len(PredictLayers) - 2; i > 0; i-- {
		if len(target) != len(PredictLayers[i+1]) {
			println("Data len Error")
		}

		// count delta
		PredictLayers[i+1] = (*c.Layers)[i].Prime(PredictLayers[i+1])
		target = (*c.Layers)[i].Prime(target)
		for j := range PredictLayers[i+1] {
			PredictLayers[i+1][j] = (PredictLayers[i+1][j] - target[j]) * 2
		}

		// loop activation

		//update bias and weight
		for j := range PredictLayers[i+1] {
			for k := range PredictLayers[i] {
				w[i][k][j] += PredictLayers[i+1][j] * PredictLayers[i][k]
				b[i][k][j] += PredictLayers[i+1][j]
			}
		}

		//fmt.Printf("weight: %.3f\n", (*(*c.Layers)[1].Neurons)[0].Weights)

		target = make([]float64, len(PredictLayers[i]))

		// next layer a loop
		for j := range PredictLayers[i] {
			target[j] = 0
			for k := range PredictLayers[i+1] {
				target[j] += target[k] * (*(*(*c.Layers)[i].Neurons)[j].Weights)[k] * PredictLayers[i+1][k]
			}
		}
	}
	return b, w, nil
}

func Accurate(predict, target []float64) float64 {
	var sum float64
	for i := range predict {
		sum += math.Abs(Cost(predict[i], target[i]))
	}
	return sum / float64(len(predict))
}

func (c *Chain) BatchAccurate(Target [][]float64, sample [][]float64) float64 {
	var accurate float64
	if len(Target) != len(sample) {
		println("Sample rate not the same, Cannot cal Accurate")
	}
	var predict []float64 = make([]float64, len(Target[0]))

	for i := 0; i < len(sample); i++ {
		predict = c.Predict(sample[i][:])
		accurate += Accurate(predict, Target[i][:])
	}

	accurate /= float64(len(sample))
	if accurate > 1 {
		accurate -= 1
	}
	return accurate
}
