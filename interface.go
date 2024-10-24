package Golang_NN

import (
	"bufio"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

type Activation func(x []float64) []float64

type Prime func(x []float64) []float64

// this interface should return pointer of a struct
type Layer interface {

	// forward prediction
	Predict(input []float64) []float64

	// for backward prediction
	Cost(target, Predicted []float64) []float64
	Exposed(thisPredicted []float64) []float64
	Update(thisPredict, ExposedNextPredict, Delta []float64)
	Delta(nextDelta, ExposedNextPredict []float64, ThisPredictLen int) []float64

	// for batch prediction
	UpdateCache(thisPredict, ExposedNextPredict, Delta [][]float64)
}

type Module struct {
	Layers []Layer
	name   string

	// for writing and reading module file
}

func NewModel() Module {
	return Module{Layers: []Layer{}}
}

func (m Module) Add(layer Layer) Module {
	m.Layers = append(m.Layers, layer)
	return m
}

func (m Module) Predict(input []float64) []float64 {
	for i := 0; i < len(m.Layers); i++ {
		input = m.Layers[i].Predict(input)
	}
	return input
}

func (m Module) Accurate(input, target []float64) float64 {
	var sum float64
	predict := m.Predict(input)
	for i := range predict {
		sum += math.Abs(Cost(predict[i], target[i]))
	}
	return sum / float64(len(predict))
}

func (m Module) Copy() Module {
	w := m
	return w
}

func (m Module) Train(input, output []float64) {
	// first, getting each layer of prediction
	var PredictData [][]float64 = make([][]float64, len(m.Layers)+1)
	PredictData[0] = input
	for i := 1; i <= len(m.Layers); i++ {
		PredictData[i] = m.Layers[i].Predict(PredictData[i-1])
	}

	// then, counting the root delta by Target Output
	delta := m.Layers[0].Cost(output, PredictData[len(m.Layers)-1])

	// layer loop from last hidden layer to first layer ,
	// i+1 == next layer , i== this layer
	for i := len(PredictData) - 2; i > 0; i-- {

		// 	using next layer cost correcting the next layer prediction??
		// for j := range PredictData[i+1] {
		// 	PredictData[i+1][j] = PredictData[i+1][j] * target[j]
		// }

		//	exposed the Prediction
		PredictData[i+1] = (m.Layers)[i+1].Exposed(PredictData[i+1])

		//	loop next layer Prediction
		m.Layers[i+1].Update(PredictData[i], PredictData[i+1], delta)

		// 	count this layer Cost
		delta = m.Layers[i].Delta(delta, PredictData[i+1], len(PredictData[i]))
	}
}

func (m Module) batchUpdate(miniBatchInput, miniBatchTarget [][]float64, sampleRate int) error {
	// pre making all PredictData for all layer by sample rate
	// predictData[layer][sample rate][data]

	if len(miniBatchInput) != len(miniBatchTarget) {
		return errors.New("dataFormate error, input len != target len")
	}
	updateLoopLength := len(miniBatchInput) / sampleRate

	// first version, need to implement the speed by this after
	//var BigBlock []float64 = make([]float64, len(m.Layers)*3*sampleRate)
	//var predictBlock []float64 = BigBlock[:len(m.Layers)*sampleRate]

	var PredictData [][][]float64 = make([][][]float64, len(m.Layers))

	var ExposedData [][][]float64 = make([][][]float64, len(m.Layers))

	var delta [][][]float64 = make([][][]float64, len(m.Layers))

	for i := range m.Layers {
		PredictData[i] = make([][]float64, sampleRate)
		delta[i] = make([][]float64, sampleRate)
		ExposedData[i] = make([][]float64, sampleRate)
	}

	var start, end int

	// loop by sample rate
	for i := 0; i <= updateLoopLength; i++ {

		start = sampleRate * i
		if i == updateLoopLength {
			end = sampleRate*i + len(miniBatchInput)%sampleRate - 1
			if start == end {
				break
			}
		} else {
			end = start + sampleRate
		}
		thisBatchTarget := miniBatchTarget[start:end]

		PredictData[0] = miniBatchInput[start:end]
		ExposedData[0] = miniBatchInput[start:end]

		// batch predict
		for j := 0; j < len(m.Layers); j++ {
			for k := 0; k < sampleRate; k++ {
				// PredictData [input -- layer end]
				PredictData[j+1][k] = m.Layers[j].Predict(PredictData[j][k])
				// ExposedData [input -- layer end]
				ExposedData[j+1][k] = m.Layers[j].Exposed(PredictData[j+1][k])
			}
		}

		// init last layer delta
		for k := 0; k < sampleRate; k++ {
			delta[len(m.Layers)-1][k] = m.Layers[len(m.Layers)-1].Cost(thisBatchTarget[k], PredictData[0][k])
		}

		for j := len(m.Layers) - 2; j > 0; j-- {
			for k := 0; k < sampleRate; k++ {
				// delta []
				delta[j][k] = m.Layers[j].Delta(delta[j+1][k], ExposedData[j][k], len(PredictData[j][i]))
			}
		}

		for j := len(m.Layers) - 2; j >= 0; j-- {
			m.Layers[j+1].UpdateCache(PredictData[j], ExposedData[j+1], delta[j])
		}

		// for j := range ExposedNextPredict {

		// 	//change bias number by cost
		// 	Delta[j] = Delta[j] * ExposedNextPredict[j]
		// 	(*d.Bias)[j] += Delta[j] * d.LearningRate

		// 	// change each weight by cost
		// 	for k := range thisPredict {
		// 		(*d.Neurons)[k][j] += Delta[j] * thisPredict[k] * d.LearningRate
		// 	}
		// }

	}
	return nil

}

func (m Module) Save(modulePath string) error {
	modulePath = filepath.Join(modulePath, m.name)
	if _, err := os.Stat(modulePath); !os.IsNotExist(err) {
		fmt.Println("Module exist, do you want to overwrite the module?")
		fmt.Println("please input Y/N:")
		input := bufio.NewScanner(os.Stdin)
		input.Scan()
		aug := input.Text()
		if aug == "N" || aug == "n" {
			return nil
		}
		// delete file
		err := os.Remove(modulePath)
		if err != nil {
			return err
		}
	}

	// create file
	folder, err := os.Create(modulePath)
	if err != nil {
		return err
	}
	defer folder.Close()
	encoder := gob.NewEncoder(folder)
	err = encoder.Encode(m)
	if err != nil {
		return err
	}

	return nil
}

// you should init new blank module before Load
func (m Module) Load(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&m)
	if err != nil {
		return err
	}

	return nil
}
