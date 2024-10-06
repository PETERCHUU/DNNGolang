package DNNGolang

type Layer interface {
	predict(input []float64) []float64
	Cost(target, Predicted []float64) LayerCost
	Prime(Predicted []float64) []float64
	Update(ThisPredict, NextPredict, Delta []float64)
	Delta(NextPredict []float64, ThisPredictLen int) []float64
}
type LayerCost interface {
	Cost()
}
