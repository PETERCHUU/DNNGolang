package function

func NormalDelta(output, target []float32) []float32 {
	for i := range output {
		output[i] = (output[i] - target[i]) * 2
	}
	return output
}
