package Mnist

import (
	"fmt"
	"nnfcgolang"
	"nnfcgolang/Mnist/FileReader"
	"nnfcgolang/function"
)

const (
	trainingDataPath  = "Sample/train/train-images.idx3-ubyte"
	trainingLabelPath = "Sample/train/train-labels.idx1-ubyte"
	testDataPath      = "Sample/train/t10k-images.idx3-ubyte"
	testLabelPath     = "Sample/train/t10k-labels.idx1-ubyte"

	sampleRate   = 1000
	learningRate = 0.15
)

func Run() {
	module := nnfcgolang.NewNetwork().FCLayer(784, 49, function.Sigmoid, learningRate).FCLayer(49, 23, function.Sigmoid, learningRate).
		FCLayer(23, 10, function.Softmax, learningRate)
	//betterModule := module
	var accurate float64
	sample := FileReader.InitSample(trainingDataPath, trainingLabelPath)
	tester := FileReader.InitSample(testDataPath, testLabelPath)

	fmt.Printf("Accurate before train: %.2f\n", calculateAccurate(&module, tester))

	// for i, v := range sample {
	// 	module.BackProp(v.Image[:], v.Label[:], learningRate)
	// 	//fmt.Printf("before weight %.2f", (*(*module.Layers)[2].Neurons)[3].Weights)
	// 	if i%1000 == 0 {
	// 		fmt.Printf("Accurate after %d train: %.4f\n", i, calculateAccurate(&module, tester))
	// 	}
	// }

	for i := 0; i < len(sample); i += sampleRate {
		sampleInput := make([][]float64, sampleRate)
		sampleTarget := make([][]float64, sampleRate)
		for j := 0; j < sampleRate; j++ {
			sampleInput[j] = sample[i+j].Image[:]
			sampleTarget[j] = sample[i+j].Label[:]
		}

		module.UpdateMiniBatch(sampleInput, sampleTarget, sampleRate, learningRate)
		thisAccurate := calculateAccurate(&module, tester)
		if thisAccurate > accurate {
			accurate = thisAccurate
			//betterModule = module
		}
		fmt.Printf("Accurate after %d train: %.4f\n", i, accurate)
	}

	//fmt.Printf("after weight %.2f", (*(*module.Layers)[2].Neurons)[3].Weights)
	fmt.Printf("Accurate after train: %.4f\n", calculateAccurate(&module, tester))
}

func calculateAccurate(module *nnfcgolang.Chain, sample []FileReader.MnstSample) float64 {
	var accurate float64
	for _, v := range sample {
		predict := module.Predict(v.Image[:])
		accurate += nnfcgolang.Accurate(predict, v.Label[:])
	}

	accurate /= float64(len(sample))
	if accurate > 1 {
		accurate -= 1
	}
	return accurate
}
