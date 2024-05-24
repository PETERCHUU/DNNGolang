package nnfcgolang

import "nnfcgolang/activation"

//intresting function for weight
func BinaryCount(n int) int {
	count := 0
	for n > 0 {
		count += n & 1
		n >>= 1
	}
	return count
}

// targetfunction
/*
     input   hidden   output
model 30 - 16 - 8 - 4 - 1
FCnetwork := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(activation.softmax)

FCnetwork.trainBy(backprop.adam)

FCnetwork.windowsize(int)


FCnetwork.train(data)

FCnetwork.test(testdata)

newwork.saveAs("model.toml")
*/

// float32 ready for CUDA
type Neuron struct {
	Len     int
	Input   float32
	Bias    float32
	Cost    float32
	Weights *[]float32 // use it as a and gate instead of float64
}

type Layer struct {
	Len          int
	Cost         float32
	Activation   activation.Activation
	LearningRate float32
	Gradient     float32
	Neurons      *[]Neuron
}

type Output struct {
	Len    int
	Cost   float32
	Output *[]float32
}

type Chain struct {
	Len     int
	Cost    float32
	Layers  *[]Layer
	Outputs *Output
}

//FCinit(30,16,activation.Sigmoid)

/*
FCnetwork := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(1,activation.softmax)
*/

func NewNetwork() Chain {
	return Chain{Len: 0, Layers: new([]Layer)}
}

func (c Chain) FCLayer(n int, next int, f activation.Activation) Chain {
	if c.Len > 0 {
		if (*c.Layers)[c.Len].Len != n {
			panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
		}
	}

	L := new([]Neuron)
	for i := 0; i < n; i++ {
		*L = append(*L, Neuron{Len: next, Input: 0, Bias: 0, Cost: 0, Weights: new([]float32)})
	}

	// add a layer
	*c.Layers = append(*c.Layers, Layer{Len: n, Neurons: L, Activation: f})
	c.Len++
	return c
}

func (c Chain) Output(n int, f activation.Activation) Chain {
	if (*c.Layers)[c.Len-1].Len != n {
		panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
	}

	// add a layer
	c.Outputs = &Output{Len: n, Output: new([]float32)}
	return c
}
