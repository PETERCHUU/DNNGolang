package dnn

import (
	"errors"
	"math"
)

// in this file, all thing go backward,

// next is the backward n-1

type LayerCost struct {
	Cost []float64
}

// Cost function for cost in layer model+1
func (d *DNN) Cost(target, Predicted []float64) *LayerCost {
	for i := range target {
		// target == cost
		target[i] = cost(Predicted[i], target[i])
	}
	return &LayerCost{Cost: target}
}

// NextCost function :
func (d *DNN) NextCost(DNNCost *LayerCost, Predicted []float64, NextLayerLen int) {
	target := make([]float64, NextLayerLen)
	// next layer a loop
	for j := 0; j < NextLayerLen; j++ {
		target[j] = 0
		for k := range Predicted {
			target[j] += DNNCost.Cost[k] * (*(*d.Neurons)[j].Weights)[k] * Predicted[k]
		}
	}
	DNNCost.Cost = target
}

// UpdateCache function
func (d *DNN) UpdateCache(delta []float64, DNNCost *LayerCost) DNN {

}

// Update function
func (d *DNN) Update(thisPredict []float64, DNNCost *LayerCost) {
	for i := range thisPredict {
		thisPredict[i] = thisPredict[i] * DNNCost.Cost[i]
	}

	// loop activation
	thisPredict = d.Prime(thisPredict)

	for j := range thisPredict {

		//change bias number by cost
		DNNCost.Cost[j] = DNNCost.Cost[j] * thisPredict[j]
		(*d.Bias)[j] += DNNCost.Cost[j] * d.LearningRate

		// change each weight by cost
		for k := range thisPredict {
			(*(*d.Neurons)[k].Weights)[j] += DNNCost.Cost[j] * thisPredict[k] * d.LearningRate
		}
	}

}

func cost(predict, target float64) float64 {
	return (target - predict) * 2
}

// z=x1y1+x2y2+....B
func (d *DNN) LayerBackProp(predicted, target []float64, learningRate float64) ([][]float64, [][][]float64, error) {

	if len(target) != len(predicted) || len(predicted) < 1 {
		return nil, nil, errors.New("dataFormate error, prediction data len != layer number")
	}

	// change target to be cost from layer
	for i := range target {
		// target == cost
		target[i] = cost(predicted[i], target[i])
	}

	//PredictLayers [][]float
	//using last layer cost calculate this layer
	for i := len(PredictLayers) - 2; i > 0; i-- {

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

		target = make([]float64, len(PredictLayers[i]))

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
		sum += math.Abs(LayerCost(predict[i], target[i]))
	}
	return sum / float64(len(predict))
}
