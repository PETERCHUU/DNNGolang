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

	sampleRate   = 1000
	learningRate = 0.7
)

func main() {
	module := nnfcgolang.NewNetwork().FCLayer(784, 49, function.ReLU, learningRate).FCLayer(49, 23, function.ReLU, learningRate).
		FCLayer(23, 10, function.Softmax, learningRate)
	sample := Sample.InitSample(trainingDataPath, trainingLabelPath)
	tester := Sample.InitSample(testDataPath, testLabelPath)

	fmt.Printf("Accurate before train: %.2f\n", calculateAccurate(&module, tester))

	// for i, v := range sample {
	// 	module.BackProp(v.Image[:], v.Label[:], learningRate)
	// 	//fmt.Printf("before weight %.2f", (*(*module.Layers)[2].Neurons)[3].Weights)
	// 	if i%1000 == 0 {
	// 		fmt.Printf("Accurate after %d train: %.2f\n", i, calculateAccurate(&module, tester))
	// 	}
	// }

	
	for i := 0; i < len(sample); i += 1000 {
		sampleInput := make([][]float32, 1000)
		sampleTarget := make([][]float32, 1000)
		for j := 0; j < 1000; j++ {
			sampleInput[j] = sample[i+j].Image[:]
			sampleTarget[j] = sample[i+j].Label[:]
		}

		module.UpdateMiniBatch(sampleInput, sampleTarget, sampleRate, learningRate)
		fmt.Printf("Accurate after %d train: %.2f\n", i, calculateAccurate(&module, tester))
	}

	//fmt.Printf("after weight %.2f", (*(*module.Layers)[2].Neurons)[3].Weights)
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
