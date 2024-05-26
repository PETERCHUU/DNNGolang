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

func SigmoidPrime(x float64) float64 {
	return x * (1 - x)
}

func TanhForward(x float64) float64 {
	return math.Tanh(x)
}

func TanhPrime(x float64) float64 {
	return 1 - x*x
}

func ReLUForward(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func SwishForward(x float64) float64 {
	return x / (1 + math.Exp(-x))
}
func SwishPrime(x float64) float64 {
	return (1 + x*(1-x))
}

func SoftmaxForward(x []float64) []float64 {
	sum := 0.0
	for i := range x {
		sum += math.Exp(x[i])
	}
	for i := range x {
		x[i] = math.Exp(x[i]) / sum
	}
	return x
}

func SoftmaxPrime(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}
