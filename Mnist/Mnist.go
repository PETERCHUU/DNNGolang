package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/PETERCHUU/Golang_NN"
	"github.com/PETERCHUU/Golang_NN/function"
)

const (
	trainingDataPath  = "Mnist/train/train-images.idx3-ubyte"
	trainingLabelPath = "Mnist/train/train-labels.idx1-ubyte"
	testDataPath      = "Mnist/train/t10k-images.idx3-ubyte"
	testLabelPath     = "Mnist/train/t10k-labels.idx1-ubyte"

	sampleRate           = 1000
	learningRate float64 = 0.15
)

type MnstSample struct {
	Label [10]float64  // nn output
	Image [784]float64 // nn input
}

func main() {
	const testDataPath = "Mnist/train/t10k-images.idx3-ubyte"
	const testLabelPath = "Mnist/train/t10k-labels.idx1-ubyte"
	Module := Run()
	filename, err := Module.Save()
	if err != nil {
		panic(err)
	}
	newModule := Golang_NN.Load(filename)
	tester := InitSample(testDataPath, testLabelPath)
	fmt.Printf("Accurate using output Module: %.4f\n", CalculateAccurate(newModule, tester))
}

// crating unit test for interface, the result should be the same
// func RunInterface() Golang_NN.Module {
// 	module := Golang_NN.NewModel().Add(dnn.NewLayer(784, 49, function.SigmoidIn, function.SigmoidOut, learningRate)).Add(dnn.NewLayer(49, 10, function.SoftmaxIn, function.SoftmaxOut, learningRate))
// 	betterModule := module.Copy()
// 	var accurate float64
// 	sample := InitSample(trainingDataPath, trainingLabelPath)
// 	tester := InitSample(testDataPath, testLabelPath)
// }

func Run() Golang_NN.Chain {
	module := Golang_NN.NewNetwork().FCLayer(784, 49, function.Sigmoid, learningRate).
		FCLayer(49, 10, function.Softmax, learningRate)
	betterModule := module.Copy()
	var accurate float64
	sample := InitSample(trainingDataPath, trainingLabelPath)
	tester := InitSample(testDataPath, testLabelPath)

	fmt.Printf("Accurate before train: %.2f\n", CalculateAccurate(&module, tester))

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
		thisAccurate := CalculateAccurate(&module, tester)
		if thisAccurate > accurate {
			accurate = thisAccurate
			betterModule = module.Copy()
		}
		fmt.Printf("Accurate after %d train: %.4f\n", i, thisAccurate)
	}

	//fmt.Printf("after weight %.2f", (*(*betterModule.Layers)[2].Neurons)[3].Weights)

	accurate = CalculateAccurate(&betterModule, tester)

	fmt.Printf("Accurate after train: %.4f\n", accurate)
	return betterModule
}

func InitSample(imageFilePath, LabelFilePath string) []MnstSample {
	// 打开文件
	imageFile, err := os.Open(imageFilePath)
	labelFile, err := os.Open(LabelFilePath)
	if err != nil {
		panic(err)
	}
	defer imageFile.Close()
	var b int32
	var length int32
	for i := 0; i < 4; i++ {
		err := binary.Read(imageFile, binary.BigEndian, &b)
		if err != nil {
			panic(err)
		}
		if i == 1 {
			length = b
		}
		//fmt.Printf("Byte value: %d\n", b)

	}
	for i := 0; i < 2; i++ {

		err := binary.Read(labelFile, binary.BigEndian, &b)
		if err != nil {
			panic(err)
		}
		//fmt.Printf("Byte value: %d\n", b)

	}
	var ImageBuffer = make([]uint8, 784)
	var LabelBuffer byte
	var Sample = make([]MnstSample, length)
	i := 0

	for {

		_, err := imageFile.Read(ImageBuffer)
		if err != nil && err != io.EOF {
			panic(err)
		}
		err = binary.Read(labelFile, binary.BigEndian, &LabelBuffer)
		if err != nil && err != io.EOF {
			panic(err)
		}
		label := [10]float64{}
		switch LabelBuffer % 10 {
		case 0:
			label[0] = 1
		case 1:
			label[1] = 1
		case 2:
			label[2] = 1
		case 3:
			label[3] = 1
		case 4:
			label[4] = 1
		case 5:
			label[5] = 1
		case 6:
			label[6] = 1
		case 7:
			label[7] = 1
		case 8:
			label[8] = 1
		case 9:
			label[9] = 1
		}

		Sample[i].Label = label

		for j := 0; j < 784; j++ {
			Sample[i].Image[j] = float64(ImageBuffer[j])
		}

		if err == io.EOF || i >= int(length)-1 {
			break
		}
		i++
	}
	return Sample
}

func CalculateAccurate(module *Golang_NN.Chain, sample []MnstSample) float64 {
	var accurate float64
	for _, v := range sample {
		predict := module.Predict(v.Image[:])
		accurate += Golang_NN.Accurate(predict, v.Label[:])
	}

	accurate /= float64(len(sample))
	if accurate > 1 {
		accurate -= 1
	}
	return accurate
}
