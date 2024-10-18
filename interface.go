package DNNGolang

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
)

type Activation func(x []float64) []float64

type Prime func(x []float64) []float64

// this interface should return pointer of a struct
type Layer interface {
	// create RNN
	Append(input int)
	AddRNN()
	IsRNN() bool

	// forward prediction
	Predict(input []float64) []float64

	// for backward prediction
	Cost(target, Predicted []float64) []float64
	Exposed(Predicted []float64) []float64
	Update(thisPredict, ExposedNextPredict, Delta []float64)
	Delta(nextDelta, NextPredict []float64, ThisPredictLen int) []float64
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

// this function should use after init last layer
func (m Module) Recur(outputLength int) {
	m.Layers[0].AddRNN()
	m.Layers[len(m.Layers)-1].AddRNN()
	m.Layers[len(m.Layers)-1].Append(outputLength)
}

func (m Module) Predict(input []float64) []float64 {
	for i := 0; i < len(m.Layers); i++ {
		input = m.Layers[i].Predict(input)
	}
	return input
}

func (m Module) Train(input, output []float64, learningRate float64) {
	// first, getting each layer of prediction
	var PredictData [][]float64
	PredictData = append(PredictData, input)
	for i := 0; i < len(m.Layers); i++ {
		// func(l *layer) predict(input []float64) []float64
		PredictData = append(PredictData, m.Layers[i].Predict(PredictData[i]))
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

		//	exposed next layer Prediction
		PredictData[i+1] = (m.Layers)[i+1].Exposed(PredictData[i+1])

		//	loop next layer Prediction
		m.Layers[i+1].Update(PredictData[i], PredictData[i+1], delta)

		// 	count this layer Cost
		//	func(l *layer) BackPropDelta (NextPredict []float64 , ThisPredictLen int) []float64
		delta = m.Layers[i].Delta(delta, PredictData[i+1], len(PredictData[i]))
	}
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
