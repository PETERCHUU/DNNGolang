package activation

import "math"

type Activation int

const (
	Sigmoid Activation = iota
	Tanh               = iota
	ReLU               = iota
	Softmax            = iota
	Swish              = iota
)

func SigmoidForward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
