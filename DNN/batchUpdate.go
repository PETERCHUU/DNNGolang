package dnn

import "github.com/PETERCHUU/Golang_NN"

//miniBatchInput, miniBatchTarget [][]float64, sampleRate int, LearningRate float64
func (d *DNN) updateCache(thisPredict, ExposedNextPredict [][]float64, Delta []float64) {

	L := make([][]float64, len(*d.Neurons))
	B := make([]float64, len(*d.Bias))
	for i := 0; i < len(*d.Neurons); i++ {
		W := make([]float64, len((*d.Neurons)[i]))
		L[i] = W
	}

	// for j := range ExposedNextPredict {

	// 	//change bias number by cost
	// 	Delta[j] = Delta[j] * ExposedNextPredict[j]
	// 	(*d.Bias)[j] += Delta[j] * d.LearningRate

	// 	// change each weight by cost
	// 	for k := range thisPredict {
	// 		(*d.Neurons)[k][j] += Delta[j] * thisPredict[k] * d.LearningRate
	// 	}
	// }

}

// UpdateCache function
// using new and old as a input
func (d *DNN) UpdateCache(od Golang_NN.Layer, thisPredict, ExposedNextPredict, Delta []float64) {

}
