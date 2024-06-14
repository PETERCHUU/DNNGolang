package main

import (
	"fmt"
	"nnfcgolang"
	"nnfcgolang/Sample"
	"nnfcgolang/function"
)

const (
	trainingDataPath  = "Sample/train/train-images.idx3-ubyte"
	trainingLabelPath = "Sample/train/train-labels.idx1-ubyte"
	testDataPath      = "Sample/train/t10k-images.idx3-ubyte"
	testLabelPath     = "Sample/train/t10k-labels.idx1-ubyte"
)

func main() {
	module := nnfcgolang.NewNetwork().FCLayer(784, 49, function.ReLU).FCLayer(49, 23, function.ReLU).
		FCLayer(23, 10, function.Softmax)
	sample := Sample.InitSample(trainingDataPath, trainingLabelPath)
	tester := Sample.InitSample(testDataPath, testLabelPath)

	fmt.Printf("Accurate before train: %.2f\n", calculateAccurate(&module, tester))

	for i, v := range sample {
		module.Train(v.Image[:], v.Label[:], 0.01)

		if i%1000 == 0 {
			fmt.Printf("Accurate after %d train: %.2f\n", i, calculateAccurate(&module, tester))
		}

	}

	fmt.Printf("Accurate after train: %.2f\n", calculateAccurate(&module, tester))
}

func calculateAccurate(module *nnfcgolang.Chain, sample []Sample.MnstSample) float32 {
	var accurate float32
	for _, v := range sample {
		predict, err := module.Predict(v.Image[:])
		if err != nil {
			panic(err)
		}
		accurate += nnfcgolang.Accurate(predict, v.Label[:])

	}
	accurate /= float32(len(sample))
	return accurate
}
