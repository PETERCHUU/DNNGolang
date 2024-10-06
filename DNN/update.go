package dnn

// in this file, all thing go backward,

// next is the backward n-1

type Cost struct {
	Cost []float64
}

// Cost function for cost in layer model+1
func (d *DNN) Cost(target, Predicted []float64) []float64 {
	for i := range target {
		// target == cost
		target[i] = cost(Predicted[i], target[i])
	}
	return target
}

// NextCost function :
func (d *DNN) Delta(nextDelta, NextPredict []float64, ThisPredictLen int) []float64 {
	target := make([]float64, ThisPredictLen)
	// next layer a loop
	for j := 0; j < ThisPredictLen; j++ {
		target[j] = 0
		for k := range NextPredict {
			target[j] += nextDelta[k] * (*(*d.Neurons)[j].Weights)[k] * NextPredict[k]
		}
	}
	return target
}

func (d *DNN) Exposed(Predicted []float64) []float64 {
	return d.Prime(Predicted)
}

// UpdateCache function
func (d *DNN) UpdateCache(delta []float64, DNNCost []float64) DNN {
	return DNN{}
}

// Update function
func (d *DNN) Update(thisPredict, ExposedNextPredict, Delta []float64) {

	for j := range ExposedNextPredict {

		//change bias number by cost
		Delta[j] = Delta[j] * ExposedNextPredict[j]
		(*d.Bias)[j] += Delta[j] * d.LearningRate

		// change each weight by cost
		for k := range thisPredict {
			(*(*d.Neurons)[k].Weights)[j] += Delta[j] * thisPredict[k] * d.LearningRate
		}
	}

}

func cost(predict, target float64) float64 {
	return (target - predict) * 2
}
