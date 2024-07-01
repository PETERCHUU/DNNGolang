package nnfcgolang

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"nnfcgolang/function"
	"os"
	"reflect"
	"time"
)

const (
	filenameF string  = "model%s.bin"
	version   float64 = 0.01
	cost      int32   = 0
)

type moduleInfo struct {
	version    float64
	createDate time.Time
	cost       float64
	length     int32
}

func (c *Chain) Save() error {

	createTime := time.Now()
	filename := fmt.Sprintf(filenameF, createTime.Format("2006-01-02_15-04"))

	// if file exist delete
	if _, err := os.Stat(filename); !os.IsNotExist(err) {
		fmt.Println("Module exist, do you want to overwrite the module?")
		fmt.Println("please input Y/N:")
		input := bufio.NewScanner(os.Stdin)
		input.Scan()
		aug := input.Text()
		if aug == "N" || aug == "n" {
			return nil
		}
		// delete file
		err := os.Remove(filename)
		if err != nil {
			return err
		}
	}

	// create file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	// Network Info
	c.writeNetworkInfo(file, createTime)

	// Layer Info
	c.writeLayerInfo(file)

	// Weight Info
	c.writeNeuronInfo(file)

	return nil

}

func (c *Chain) writeNetworkInfo(file *os.File, date time.Time) {
	info := moduleInfo{
		version:    0.1,
		createDate: date,
		cost:       0,
		length:     int32(len(*c.Layers)),
	}
	writeBin(file, info)
}

func (c *Chain) writeLayerInfo(file *os.File) {
	for _, l := range *c.Layers {
		binary.Write(file, binary.BigEndian, len(*l.Neurons)) // this length
		binary.Write(file, binary.BigEndian, len(*l.Bias))    // next length
		binary.Write(file, binary.BigEndian, function.GetEnum(l.Activation))
		binary.Write(file, binary.BigEndian, l.LearningRate)
	}
}

// next layer length
func (c *Chain) writeNeuronInfo(file *os.File) {
	for _, l := range *c.Layers {
		for _, n := range *l.Bias {
			binary.Write(file, binary.BigEndian, n)
		}
		for _, n := range *l.Neurons {
			for _, w := range *n.Weights {
				binary.Write(file, binary.BigEndian, w)
			}
		}
	}
}

// file, err := os.Create("model.bin")
// if err != nil {
// 	panic(err)
// }
// defer file.Close()

func writeBin(file *os.File, field interface{}) {
	value := reflect.ValueOf(field)
	switch value.Kind() {
	case reflect.Struct:
		// loop and write bin
		for i := 0; i < value.NumField(); i++ {
			binary.Write(file, binary.BigEndian, value.Field(i))
		}
	case reflect.Array:
		fallthrough
	case reflect.Ptr:
		fallthrough
	case reflect.Slice:
		for i := 0; i < value.Elem().Len(); i++ {
			binary.Write(file, binary.BigEndian, value.Field(i))
		}
	default:
		binary.Write(file, binary.BigEndian, value)
	}
}

func Load(path string) *Chain {
	file, err := os.Open(path)
	if err != nil {
		print(err)
		return nil
	}
	defer file.Close()
	var info moduleInfo
	binary.Read(file, binary.BigEndian, &info.version)
	binary.Read(file, binary.BigEndian, &info.createDate)
	binary.Read(file, binary.BigEndian, &info.cost)
	binary.Read(file, binary.BigEndian, &info.length)
	switch info.version {
	case 0.1:
		fallthrough
	default:
		return readFile(file, info.length, info.cost)
	}

}

func readFile(file *os.File, length int32, cost float64) *Chain {
	layer := make([]FCLayer, length)
	var Chain Chain
	Chain.Layers = &layer
	Chain.Cost = Cost
	var LayerType int32

	for i := 0; i < int(length); i++ {
		binary.Read(file, binary.BigEndian, &LayerType)
		switch LayerType {
		default:
			var this int
			var next int
			var activate int32
			var rate float64
			binary.Read(file, binary.BigEndian, &this)
			binary.Read(file, binary.BigEndian, &next)
			binary.Read(file, binary.BigEndian, &activate)
			binary.Read(file, binary.BigEndian, &rate)
			Chain.FCLayer(this, next, function.Activation(activate), rate)
		}
	}
	// read bias
	for i := 0; i < len(*Chain.Layers); i++ {
		for j := 0; j < len(*(*Chain.Layers)[i].Bias); j++ {
			binary.Read(file, binary.BigEndian, &(*(*Chain.Layers)[i].Bias)[j])
		}
		for j := 0; j < len(*(*Chain.Layers)[i].Neurons); j++ {
			for k := 0; k < len(*(*(*Chain.Layers)[i].Neurons)[j].Weights); k++ {
				binary.Read(file, binary.BigEndian, &(*(*(*Chain.Layers)[i].Neurons)[j].Weights)[k])
			}
		}
	}
	return &Chain
}
