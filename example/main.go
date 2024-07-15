package main

import (
	"math/rand"

	"github.com/PETERCHUU/DNNGolang"
	"github.com/PETERCHUU/DNNGolang/function"
)

const learningRate float64 = 0.15

func main() {
	//making a 700 floating point input , 1 layer of hidden with 50 plating point, and final 10 point of output
	module := DNNGolang.NewNetwork().FCLayer(700, 50, function.Sigmoid, learningRate).
		FCLayer(50, 10, function.Softmax, learningRate)

	// get a 700 length 2D array for input use
	array := make([]float64, 700)
	for i := 0; i < len(array); i++ {
		array[i] = rand.Float64()
	}
	answer := module.Predict(array)
	standardAnswer := 0.0
	for _, i := range answer {
		if i > standardAnswer {
			standardAnswer = i
		}
	}

	println("this is the answer", standardAnswer)
}
