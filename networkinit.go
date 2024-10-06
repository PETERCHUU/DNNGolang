package DNNGolang

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

// this layer , next layer, non-linear function, training rate
func (c Chain) FCLayer(n int32, next int32, f function.Activation, rate float64) Chain {

	// forward, backward
	I, O := function.ActivationFunc(f)
	println("layer init")
	//println(f)
	L := make([]Neuron, n)
	B := make([]float64, next)
	if len(*c.Layers) == 0 {
		for i := 0; i < int(n); i++ {
			W := make([]float64, next)
			L[i] = Neuron{Weights: &W}
		}
	} else {
		if (*c.Layers)[len(*c.Layers)-1].NNtype == RNN {
			panic("RNN must be the last layer")
		}
		if len(*(*(*c.Layers)[len(*c.Layers)-1].Neurons)[0].Weights) != int(n) {
			panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
		}
		for i := 0; i < int(n); i++ {
			W := make([]float64, next)
			L[i] = Neuron{Weights: &W}
		}
	}

	// add a layer
	*c.Layers = append(*c.Layers, FCLayer{LearningRate: rate, Neurons: &L, Activation: I, Prime: O, Bias: &B, ActivateEnum: f})

	return c

}

func (c Chain) RNN() Chain {

	for i := 0; i < len(*(*c.Layers)[len(*c.Layers)-1].Bias); i++ {
		W := make([]float64, len(*(*c.Layers)[0].Bias))
		*(*c.Layers)[0].Neurons = append(*(*c.Layers)[0].Neurons, Neuron{Weights: &W})
	}

	(*c.Layers)[len(*c.Layers)-1].NNtype = RNN
	println("RNN created")
	return c
}

func (c Chain) Copy() Chain {
	L := make([]FCLayer, len(*c.Layers))
	for i := 0; i < len(*c.Layers); i++ {
		N := make([]Neuron, len(*(*c.Layers)[i].Neurons))
		B := make([]float64, len(*(*c.Layers)[i].Bias))
		for j := 0; j < len(*(*c.Layers)[i].Bias); j++ {
			B[j] = (*(*c.Layers)[i].Bias)[j]
		}
		for j := 0; j < len(*(*c.Layers)[i].Neurons); j++ {
			W := make([]float64, len(*(*(*c.Layers)[i].Neurons)[j].Weights))
			for k := 0; k < len(*(*(*c.Layers)[i].Neurons)[j].Weights); k++ {
				W[k] = (*(*(*c.Layers)[i].Neurons)[j].Weights)[k]
			}
			N[j] = Neuron{Weights: &W}
		}
		L[i] = FCLayer{LearningRate: (*c.Layers)[i].LearningRate, Neurons: &N, Activation: (*c.Layers)[i].Activation, Prime: (*c.Layers)[i].Prime, Bias: &B, ActivateEnum: (*c.Layers)[i].ActivateEnum}
	}
	return Chain{Layers: &L}
}

func (c *Chain) Random() {
	for i := 0; i < len(*c.Layers); i++ {
		for j := 0; j < len(*(*c.Layers)[i].Neurons); j++ {
			for k := 0; k < len(*(*(*c.Layers)[i].Neurons)[j].Weights); k++ {
				(*(*(*c.Layers)[i].Neurons)[j].Weights)[k] = rand.Float64()
			}
		}
		for j := 0; j < len(*(*c.Layers)[i].Bias); j++ {
			(*(*c.Layers)[i].Bias)[j] = rand.Float64()
		}
	}

}

// RNN
/*
	model := nnfcgolang.NewNetwork().RNN(7,32,169,activation.Sigmoid,activation.Softmax)
*/
