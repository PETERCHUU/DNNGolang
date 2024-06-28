package nnfcgolang

type nn interface {
	Predict(data [][]float32) ([][]float32, error)
}
