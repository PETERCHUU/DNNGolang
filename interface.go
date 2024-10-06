package DNNGolang

type Layer interface {
	predict(input []float64) []float64
	Cost(target, Predicted []float64) []float64
	Exposed(Predicted []float64) []float64
	Update(thisPredict, ExposedNextPredict, Delta []float64)
	Delta(nextDelta, NextPredict []float64, ThisPredictLen int) []float64
}
