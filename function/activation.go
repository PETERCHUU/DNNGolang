package function

import (
	"math"
	"reflect"
)

type Activation int

const (
	Sigmoid Activation = iota
	Tanh
	ReLU
	Swish
	Softmax
)

func GetEnum(function func(x []float64) []float64) int32 {
	if reflect.DeepEqual(function, SigmoidIn) {
		return 0
	}
	if reflect.DeepEqual(function, TanhIn) {
		return 1
	}
	if reflect.DeepEqual(function, ReLUIn) {
		return 2
	}
	if reflect.DeepEqual(function, SwishIn) {
		return 3
	}
	if reflect.DeepEqual(function, SoftmaxIn) {
		return 4
	}
	return -1

}

func ActivationFunc(activation Activation) (func(x []float64) []float64, func(x []float64) []float64) {
	switch activation {
	case Sigmoid:
		return SigmoidIn, SigmoidOut
	case Tanh:
		return TanhIn, TanhOut
	case ReLU:
		return ReLUIn, ReLUOut
	case Swish:
		return SwishIn, SwishOut
	case Softmax:
		return SoftmaxIn, SoftmaxOut
	}
	return nil, nil

}

func SigmoidIn(x []float64) []float64 {
	for i := range x {
		x[i] = 1 / (1 + math.Exp(-x[i]))

	}
	return x
}

func SigmoidOut(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}

func TanhIn(x []float64) []float64 {
	for i := range x {
		x[i] = math.Tanh(x[i])
	}
	return x
}

func TanhOut(x []float64) []float64 {
	for i := range x {
		x[i] = 1 - x[i]*x[i]
	}
	return x
}

func ReLUIn(x []float64) []float64 {
	for i := range x {
		if x[i] < 0 || math.IsNaN(x[i]) {
			x[i] = 0
		}
	}
	return x
}

func ReLUOut(x []float64) []float64 {
	for i := range x {
		if x[i] > 0 {
			x[i] = 1
		} else {
			x[i] = 0
		}
	}
	return x
}

func SwishIn(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] / (1 + math.Exp(-x[i]))

	}
	return x
}
func SwishOut(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] + (1-x[i])/(1+math.Exp(-x[i]))

	}
	return x
}

func SoftmaxIn(x []float64) []float64 {
	var sum float64
	for i := range x {
		sum += x[i]
	}
	for i := range x {
		x[i] = math.Exp(x[i]) / sum
		if math.IsNaN(x[i]) {
			x[i] = 0
		}

	}
	return x
}

func SoftmaxOut(x []float64) []float64 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}
