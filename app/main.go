package main

import (
	"fmt"

	"github.com/PETERCHUU/DNNGolang"
	"github.com/PETERCHUU/DNNGolang/Mnist"
	"github.com/PETERCHUU/DNNGolang/Mnist/FileReader"
)

func main() {
	const testDataPath = "Mnist/train/t10k-images.idx3-ubyte"
	const testLabelPath = "Mnist/train/t10k-labels.idx1-ubyte"
	Module := Mnist.Run()
	filename, err := Module.Save()
	if err != nil {
		panic(err)
	}
	newModule := DNNGolang.Load(filename)
	tester := FileReader.InitSample(testDataPath, testLabelPath)
	fmt.Printf("Accurate using output Module: %.4f\n", Mnist.CalculateAccurate(newModule, tester))
}
