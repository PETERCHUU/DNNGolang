package dnn

import (
	"math/rand"

	"github.com/PETERCHUU/DNNGolang/function"
)

type NNType int

const (
	FC NNType = iota
	RNN
)

// intresting function for weight
func BinaryCount(n int) int {
	count := 0
	for n > 0 {
		count += n & 1
		n >>= 1
	}
	return count
}

// target function
/*
     input   hidden   output
model 30 - 16 - 8 - 4 - 1
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(activation.softmax)




model.saveAs("model.toml")
*/
// input have next layer w and b

// Neuron is a struct of neuron,
// float32 ready for CUDA
type Neuron struct {
	Weights *[]float64 // len is number of next layer
}

// FCLayer is a struct of layer
type DNN struct {
	NNtype       NNType
	LearningRate float64
	Bias         *[]float64 // len is number of next layer
	Neurons      *[]Neuron  // len is number of this layer
	ActivateEnum function.Activation
	Activation   func(x []float64) []float64
	Prime        func(x []float64) []float64
}

//FCinit(30,16,activation.Sigmoid)

/*
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.softmax)

the len(model):=[16,8,4]

*/

// this layer , next layer, non-linear function, training rate
func NewLayer(n int32, next int32, f function.Activation, rate float64) DNN {

	// forward, backward
	I, O := function.ActivationFunc(f)
	println("layer init")
	//println(f)
	L := make([]Neuron, n)
	B := make([]float64, next)

	for i := 0; i < int(n); i++ {
		W := make([]float64, next)
		L[i] = Neuron{Weights: &W}
	}

	// add a layer
	return DNN{LearningRate: rate, Neurons: &L, Activation: I, Prime: O, Bias: &B, ActivateEnum: f}

}

func (d *DNN) Copy() *DNN {
	N := make([]Neuron, len(*d.Neurons))
	B := make([]float64, len(*d.Bias))
	for j := 0; j < len(*d.Bias); j++ {
		B[j] = (*d.Bias)[j]
	}
	for j := 0; j < len(*d.Neurons); j++ {
		W := make([]float64, len(*(*d.Neurons)[j].Weights))
		for k := 0; k < len(*(*d.Neurons)[j].Weights); k++ {
			W[k] = (*(*d.Neurons)[j].Weights)[k]
		}
		N[j] = Neuron{Weights: &W}
	}
	return &DNN{LearningRate: d.LearningRate, Neurons: &N, Activation: d.Activation, Prime: d.Prime, Bias: &B, ActivateEnum: d.ActivateEnum}
}

func (d *DNN) Random() {
	for j := 0; j < len(*d.Neurons); j++ {
		for k := 0; k < len(*(*d.Neurons)[j].Weights); k++ {
			(*(*d.Neurons)[j].Weights)[k] = rand.Float64()
		}
	}
	for j := 0; j < len(*d.Bias); j++ {
		(*d.Bias)[j] = rand.Float64()
	}
}

// RNN
/*
	model := nnfcgolang.NewNetwork().RNN(7,32,169,activation.Sigmoid,activation.Softmax)
*/
