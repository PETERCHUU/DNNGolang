package nnfcgolang_test

import (
	"fmt"
	"nnfcgolang"
	"nnfcgolang/activation"
	"testing"
)

func TestNewNetwork(T *testing.T) {
	FCnetwork := nnfcgolang.NewNetwork().FCLayer(16, 8, activation.Sigmoid).
		FCLayer(8, 4, activation.Sigmoid).FCLayer(4, 1, activation.Sigmoid).
		Output(1, activation.Softmax)

	for i, v := range *FCnetwork.Layers {
		for j, n := range *v.Neurons {
			T.Log(i, j, n)
		}
	}
	//FCnetwork.trainBy(backprop.adam)
	//FCnetwork.windowsize(int)
	//FCnetwork.train(data)
	//FCnetwork.test(testdata)
	//newwork.saveAs("model.toml")
}

func BenchmarkNewNetwork(T *testing.B) {
	FCnetwork := nnfcgolang.NewNetwork().FCLayer(16, 8, activation.Sigmoid).
		FCLayer(8, 4, activation.Sigmoid).FCLayer(4, 1, activation.Sigmoid).
		Output(1, activation.Softmax)

	for i, v := range *FCnetwork.Layers {
		for j, n := range *v.Neurons {
			fmt.Println(i, j, n)
		}
	}
	//FCnetwork.trainBy(backprop.adam)
	//FCnetwork.windowsize(int)
	//FCnetwork.train(data)
	//FCnetwork.test(testdata)
	//newwork.saveAs("model.toml")
}
