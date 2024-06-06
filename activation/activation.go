package activation

import "math"

type Activation int

const (
	Sigmoid Activation = iota
	Tanh
	ReLU
	Swish
	Softmax
)

func ActivationFunc(activation Activation) (func(x []float32) []float32, func(x []float32) []float32) {
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

func SigmoidIn(x []float32) []float32 {
	for i := range x {
		x[i] = 1 / (1 + float32(math.Exp(float64(-x[i]))))

	}
	return x
}

func SigmoidOut(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}

func TanhIn(x []float32) []float32 {
	for i := range x {
		x[i] = float32(math.Tanh(float64(x[i])))
	}
	return x
}

func TanhOut(x []float32) []float32 {
	for i := range x {
		x[i] = 1 - x[i]*x[i]
	}
	return x
}

func ReLUIn(x []float32) []float32 {
	for i := range x {
		if x[i] > 0 {
			x[i] = x[i]
		} else {
			x[i] = 0
		}
	}
	return x
}

func ReLUOut(x []float32) []float32 {
	for i := range x {
		if x[i] > 0 {
			x[i] = 1
		} else {
			x[i] = 0
		}
	}
	return x
}

func SwishIn(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] / (1 + float32(math.Exp(float64(-x[i]))))

	}
	return x
}
func SwishOut(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] + (1-x[i])/(1+float32(math.Exp(float64(-x[i]))))

	}
	return x
}

func SoftmaxIn(x []float32) []float32 {
	var sum float32
	for i := range x {
		sum += float32(math.Exp(float64(x[i])))
	}
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i]))) / sum
	}
	return x
}

func SoftmaxOut(x []float32) []float32 {
	for i := range x {
		x[i] = x[i] * (1 - x[i])
	}
	return x
}
