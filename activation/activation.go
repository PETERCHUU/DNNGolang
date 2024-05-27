package activation

import "math"

func Sigmoid(x []float64) []float64 {
	for i := range x {
		x[i] = 1 / (1 + math.Exp(-x[i]))

	}
	return x
}

func SigmoidPrime(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}

func Tanh(x []float64) []float64 {
	for i := range x {
		x[i] = math.Tanh(x[i])
	}
	return x
}

func TanhPrime(x []float64) []float64 {
	for i := range x {
		x[i] = 1 - x[i]*x[i]
	}
	return x
}

func ReLU(x []float64) []float64 {
	for i := range x {
		if x[i] > 0 {
			x[i] = x[i]
		} else {
			x[i] = 0
		}
	}
	return x
}

func ReLUPrime(x []float64) []float64 {
	for i := range x {
		if x[i] > 0 {
			x[i] = 1
		} else {
			x[i] = 0
		}
	}
	return x
}

func Swish(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] / (1 + math.Exp(-x[i]))

	}
	return x
}
func SwishPrime(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] + (1-x[i])/(1+math.Exp(-x[i]))

	}
	return x
}

func Softmax(x []float64) []float64 {
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
