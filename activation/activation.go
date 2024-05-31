package activation

import "math"

func Sigmoid(x []float32) []float32 {
	for i := range x {
		x[i] = 1 / (1 + float32(math.Exp(float64(-x[i]))))

	}
	return x
}

func SigmoidPrime(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}

func Tanh(x []float32) []float32 {
	for i := range x {
		x[i] = float32(math.Tanh(float64(x[i])))
	}
	return x
}

func TanhPrime(x []float32) []float32 {
	for i := range x {
		x[i] = 1 - x[i]*x[i]
	}
	return x
}

func ReLU(x []float32) []float32 {
	for i := range x {
		if x[i] > 0 {
			x[i] = x[i]
		} else {
			x[i] = 0
		}
	}
	return x
}

func ReLUPrime(x []float32) []float32 {
	for i := range x {
		if x[i] > 0 {
			x[i] = 1
		} else {
			x[i] = 0
		}
	}
	return x
}

func Swish(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] / (1 + float32(math.Exp(float64(-x[i]))))

	}
	return x
}
func SwishPrime(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] + (1-x[i])/(1+float32(math.Exp(float64(-x[i]))))

	}
	return x
}

func Softmax(x []float32) []float32 {
	var sum float32
	for i := range x {
		sum += float32(math.Exp(float64(x[i])))
	}
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i]))) / sum
	}
	return x
}

func SoftmaxPrime(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}
