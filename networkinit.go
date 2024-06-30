package nnfcgolang

import (
	"nnfcgolang/function"
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
type FCLayer struct {
	NNtype       NNType
	LearningRate float64
	Bias         *[]float64 // len is number of next layer
	Neurons      *[]Neuron  // len is number of this layer
	ActivateEnum function.Activation
	Activation   func(x []float64) []float64
	Prime        func(x []float64) []float64
}

// Chain is a struct of model
type Chain struct {
	Cost   func(predict, target float64) float64
	Layers *[]FCLayer
	Cache  *[]FCLayer
	input  *[][]float64
}

//FCinit(30,16,activation.Sigmoid)

/*
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.softmax)

the len(model):=[16,8,4]

*/

func NewNetwork() Chain {
	return Chain{Layers: new([]FCLayer)}
}

func (c Chain) FCLayer(n int, next int, f function.Activation, rate float64) Chain {
	I, O := function.ActivationFunc(f)
	L := make([]Neuron, n)
	B := make([]float64, next)

	if len(*c.Layers) == 0 {
		for i := 0; i < n; i++ {
			W := make([]float64, next)
			L[i] = Neuron{Weights: &W}
		}
	} else {
		if (*c.Layers)[len(*c.Layers)-1].NNtype == RNN {
			panic("RNN must be the last layer")
		}
		if len(*(*(*c.Layers)[len(*c.Layers)-1].Neurons)[0].Weights) != n {
			panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
		}
		for i := 0; i < n; i++ {
			W := make([]float64, next)
			L[i] = Neuron{Weights: &W}
		}
	}

	// add a layer
	*c.Layers = append(*c.Layers, FCLayer{LearningRate: rate, Neurons: &L, Activation: I, Prime: O, Bias: &B, ActivateEnum: f})
	return c
}

func (c Chain) RNN() Chain {
	(*c.Layers)[len(*c.Layers)-1].NNtype = RNN
	return c
}

// RNN
/*
	model := nnfcgolang.NewNetwork().RNN(7,32,169,activation.Sigmoid,activation.Softmax)
*/
