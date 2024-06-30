package nnfcgolang

type nn interface {
	Predict(data [][]float64) ([][]float64, error)
}
