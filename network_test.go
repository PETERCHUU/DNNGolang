package DNNGolang_test

import (
	"fmt"
	"testing"

	"github.com/PETERCHUU/DNNGolang"
	"github.com/PETERCHUU/DNNGolang/function"
)

const learningRate = 0.1

func TestNewNetwork(T *testing.T) {
	FCnetwork := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).
		FCLayer(8, 4, function.Sigmoid, learningRate).FCLayer(4, 1, function.Softmax, learningRate)

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
	FCnetwork := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).
		FCLayer(8, 4, function.Sigmoid, learningRate).FCLayer(4, 1, function.Softmax, learningRate)
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

func TestFCLayer(t *testing.T) {
	nn := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate)
	println(len(*nn.Layers))
	t.Log("haha")
}

func TestRNN(t *testing.T) {
	nn := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).RNN()
	println(len(*nn.Layers))
	t.Log("RNN created")

}

func TestPredict(t *testing.T) {
	FCnetwork := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).
		FCLayer(8, 4, function.Sigmoid, learningRate).FCLayer(4, 1, function.Softmax, learningRate)
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	output := FCnetwork.Predict(data)
	fmt.Println()
	for i := range output {
		t.Log(output[i])
	}
}

func TestRNNpredict(t *testing.T) {
	nn := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).
		FCLayer(8, 4, function.Sigmoid, learningRate).FCLayer(4, 1, function.Softmax, learningRate).RNN()
	twoDArray := make([][]float64, 10)
	for i := 0; i < len(twoDArray); i++ {
		println(twoDArray)
		twoDArray[i] = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	}
	nn.RNNPredict(twoDArray)

}

func BenchmarkPredict(t *testing.B) {
	FCnetwork := DNNGolang.NewNetwork().FCLayer(16, 8, function.Sigmoid, learningRate).
		FCLayer(8, 4, function.Sigmoid, learningRate).FCLayer(4, 1, function.Softmax, learningRate)
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	output := FCnetwork.Predict(data)
	fmt.Println()
	for i := range output {
		t.Log(output[i])
	}
}
