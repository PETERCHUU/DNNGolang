package main

import (
	"fmt"
	"nnfcgolang"
	"nnfcgolang/Mnist"
	"nnfcgolang/Mnist/FileReader"
)

func main() {
	const testDataPath = "Mnist/train/t10k-images.idx3-ubyte"
	const testLabelPath = "Mnist/train/t10k-labels.idx1-ubyte"
	// Module := Mnist.Run()
	// err := Module.Save()
	// if err != nil {
	// 	panic(err)
	// }
	newModule := nnfcgolang.Load("model2024-07-01_10-09.bin")
	tester := FileReader.InitSample(testDataPath, testLabelPath)
	fmt.Printf("Accurate using output Module: %.4f\n", Mnist.CalculateAccurate(newModule, tester))
}
