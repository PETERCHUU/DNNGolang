package function

func NormalDelta(output, target []float64) []float64 {
	for i := range output {
		output[i] = (output[i] - target[i]) * 2
	}
	return output
}
