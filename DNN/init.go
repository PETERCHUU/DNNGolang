package dnn

import (
	"math/rand"

	"github.com/PETERCHUU/Golang_NN"
)

type NNType int

// FCLayer is a struct of layer
type DNN struct {
	RNN          bool
	LearningRate float64
	Bias         *[]float64   // len is number of next layer
	Neurons      *[][]float64 // len is number of this layer
	Activation   func(x []float64) []float64
	Prime        func(x []float64) []float64
}

// this layer , next layer, non-linear function, training rate
func NewLayer(n int32, nextN int32, activation Golang_NN.Activation, prime Golang_NN.Prime, rate float64) Golang_NN.Layer {

	println("layer init")
	//println(f)
	L := make([][]float64, n)
	B := make([]float64, nextN)
	for i := 0; i < int(n); i++ {
		W := make([]float64, nextN)
		L[i] = W
	}

	// add a layer
	return &DNN{LearningRate: rate, Neurons: &L, Activation: activation, Prime: prime, Bias: &B}
}

func (d *DNN) Append(input int) {
	nextN := len(*d.Bias)
	for i := 0; i < input; i++ {
		w := make([]float64, nextN)
		*d.Neurons = append(*d.Neurons, w)
	}
}

func (d *DNN) Random() {
	for j := 0; j < len(*d.Neurons); j++ {
		for k := 0; k < len((*d.Neurons)[j]); k++ {
			(*d.Neurons)[j][k] = rand.Float64()
		}
	}
	for j := 0; j < len(*d.Bias); j++ {
		(*d.Bias)[j] = rand.Float64()
	}
}
