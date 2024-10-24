package dnn

//[this sample][layer]
func (d *DNN) UpdateCache(thisPredict, ExposedNextPredict, Delta [][]float64) {

	for i := range ExposedNextPredict {
		for j := range ExposedNextPredict[i] {

			//change bias number by cost
			Delta[i][j] = Delta[i][j] * ExposedNextPredict[i][j]
			(*d.Bias)[j] += Delta[i][j] * d.LearningRate

			// change each weight by cost
			for k := range thisPredict[i] {
				(*d.Neurons)[k][j] += Delta[i][j] * thisPredict[i][k] * d.LearningRate
			}
		}
	}

}
