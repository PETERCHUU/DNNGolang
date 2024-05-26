package nnfcgolang

import "errors"

// target function
/*
	output,err:=network.Predict(data)
*/

func (c Chain) Predict(data []float32) ([]float32, error) {
	//input layer
	if len(data) != (*c.Layers)[0].Len {
		return nil, errors.New("data length not match input layer")
	}
	//hidden layer
	for i, v := range *c.Layers {
		if i == 0 {
			for j, n := range *v.Neurons {
				(*v.Neurons)[j].Input = data[j]
			}
		} else {
			for j, n := range *v.Neurons {
				(*v.Neurons)[j].Input = 0
				for k, m := range *(*c.Layers)[i-1].Neurons {
					(*v.Neurons)[j].Input += (*(*c.Layers)[i-1].Neurons)[k].Input * (*(*(*c.Layers)[i-1].Neurons)[k].Weights)[j]
				}
				(*v.Neurons)[j].Input += (*v.Neurons)[j].Bias
			}
		}
		//output layer
		if i == len(*c.Layers)-1 {
			output := make([]float32, len(*v.Neurons))
			for j, n := range *v.Neurons {
				output[j] = (*v.Neurons)[j].Input
			}
			return output, nil
		}
	}
	return nil, errors.New("predict error")
}
