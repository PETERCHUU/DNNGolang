package nnfcgolang

import (
	"nnfcgolang/activation"
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

// Neuron is a struct of neuron,
// float32 ready for CUDA
type Neuron struct {
	Weights  *[]float32 // len is number of next layer
	Gradient *[]float32 // same
}

// FCLayer is a struct of layer
type FCLayer struct {
	LearningRate float32
	Bias         *[]float32 // len is number of next layer
	Neurons      *[]Neuron  // len is number of this layer
	Activation   func(x []float32) []float32
	Prime        func(x []float32) []float32
}

// Chain is a struct of model
type Chain struct {
	Cost   float32
	Layers *[]FCLayer
}

//FCinit(30,16,activation.Sigmoid)

/*
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(1,activation.softmax)
*/

func NewNetwork() Chain {
	return Chain{Layers: new([]FCLayer)}
}

func (c Chain) FCLayer(n int, next int, f activation.Activation) Chain {
	I, O := activation.ActivationFunc(f)
	L := make([]Neuron, n)
	B := make([]float32, next)

	if len(*c.Layers) == 0 {
		for i := 0; i < n; i++ {
			W := make([]float32, next)
			L[i] = Neuron{Weights: &W}
		}
	} else {
		if len(*(*(*c.Layers)[len(*c.Layers)-1].Neurons)[0].Weights) != n {
			panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
		}
		for i := 0; i < n; i++ {
			W := make([]float32, next)
			L[i] = Neuron{Weights: &W}
		}
	}

	// add a layer
	*c.Layers = append(*c.Layers, FCLayer{Neurons: &L, Activation: I, Prime: O, Bias: &B})
	return c
}
