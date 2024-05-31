package nnfcgolang

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
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(activation.softmax)



model.trainBy(backProp.adam)

model.windowSize(int)

model.train(data)

model.test(testdata)

model.saveAs("model.toml")
*/

// float32 ready for CUDA
type Neuron struct {
	Cost    float32
	Weights *[]float32 // len is number of next layer
}

type FCLayer struct {
	NextLen      int
	Cost         float32
	LearningRate float32
	Gradient     float32
	Bias         *[]float32 // len is number of next layer
	Neurons      *[]Neuron  // len is number of this layer
	Activation   func(x []float32) []float32
}

type Chain struct {
	Cost    float32
	Layers  *[]FCLayer
	Outputs func(x []float32) []float32
}

//FCinit(30,16,activation.Sigmoid)

/*
model := nnfcgolang.NewNetwork().FCLayer(16,8,activation.Sigmoid).
FCLayer(8,4,activation.Sigmoid).FCLayer(4,1,activation.Sigmoid).Output(1,activation.softmax)
*/

func NewNetwork() Chain {
	return Chain{Layers: new([]FCLayer)}
}

func (c Chain) FCLayer(n int, next int, f func(x []float32) []float32) Chain {
	L := make([]Neuron, n)
	B := make([]float32, next)

	if len(*c.Layers) == 0 {
		for i := 0; i < n; i++ {
			W := make([]float32, next)
			L[i] = Neuron{Cost: 0, Weights: &W}
		}
	} else {
		if len(*(*(*c.Layers)[len(*c.Layers)-1].Neurons)[0].Weights) != n {
			panic("The number of neurons in the previous layer does not match the number of neurons in the current layer")
		}
		for i := 0; i < n; i++ {
			W := make([]float32, next)
			L[i] = Neuron{Cost: 0, Weights: &W}
		}
	}

	// add a layer
	*c.Layers = append(*c.Layers, FCLayer{NextLen: next, Neurons: &L, Activation: f, Bias: &B})
	return c
}
