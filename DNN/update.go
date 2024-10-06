package dnn

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
	return DNN{}
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
