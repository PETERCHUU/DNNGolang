package DNNGolang

type Layer interface {
	Predict(input []float64) []float64
	Cost(target, Predicted []float64) []float64
	Exposed(Predicted []float64) []float64
	Update(thisPredict, ExposedNextPredict, Delta []float64)
	Delta(nextDelta, NextPredict []float64, ThisPredictLen int) []float64
	SaveLayer()
	ReadLayer()
}

type Activation func(x []float64) []float64

type Prime func(x []float64) []float64
