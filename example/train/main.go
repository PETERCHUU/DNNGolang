package main

import (
	"math/rand"

	"github.com/PETERCHUU/DNNGolang"
	"github.com/PETERCHUU/DNNGolang/function"
)

const learningRate float64 = 0.15

// for training, please make a 3D array
func main() {
	//making a 700 floating point input , 1 layer of hidden with 50 plating point, and final 10 point of output
	module := DNNGolang.NewNetwork().FCLayer(700, 50, function.Sigmoid, learningRate).
		FCLayer(50, 10, function.Softmax, learningRate)

	Inputs := make([][]float64, 1000)

	for i := 0; i < len(Inputs); i++ {
		// get a 700 length 2D array for input use
		array := make([]float64, 700)
		for j := 0; j < len(array); j++ {
			array[j] = rand.Float64()
		}
	}

	Target := make([][]float64, 1000)
	for i := 0; i < len(Inputs); i++ {
		// get a 700 length 2D array for input use
		array := make([]float64, 10)
		for j := 0; j < len(array); j++ {
			array[j] = rand.Float64()
		}
	}

	module.UpdateMiniBatch(Inputs, Target, 100, learningRate)

	ForPredict := make([]float64, 700)
	for j := 0; j < len(ForPredict); j++ {
		ForPredict[j] = rand.Float64()
	}

	answer := module.Predict(ForPredict)
	standardAnswer := 0.0
	for _, i := range answer {
		if i > standardAnswer {
			standardAnswer = i
		}
	}

	println("this is the answer", standardAnswer)

}
