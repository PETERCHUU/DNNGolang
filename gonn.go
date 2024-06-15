package nnfcgolang

import (
	"encoding/binary"
	"fmt"
	"nnfcgolang/function"
	"os"
	"time"
)

const (
	filenameF string  = "model%s.bin"
	version   float32 = 0.01
	cost      int32   = 0
)

func (c *Chain) Save() error {
	createTime := time.Now()
	filename := fmt.Sprintf(filenameF, createTime.Format("2006-01-02_15-04"))

	if _, err := os.Stat(filename); !os.IsNotExist(err) {

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
	c.writeNetworkInfo(file)

	// Layer Info
	c.writeLayerInfo(file)

	// Weight Info

	return nil

}

func (c *Chain) writeNetworkInfo(file *os.File) {
	binary.Write(file, binary.BigEndian, version)
	binary.Write(file, binary.BigEndian, cost)
	binary.Write(file, binary.BigEndian, int32(len(*c.Layers)))
}

func (c *Chain) writeLayerInfo(file *os.File) {
	for _, l := range *c.Layers {
		binary.Write(file, binary.BigEndian, int32(len(*l.Neurons))) // this length
		binary.Write(file, binary.BigEndian, int32(len(*l.Bias)))    // next length
		binary.Write(file, binary.BigEndian, function.GetEnum(l.Activation))
		binary.Write(file, binary.BigEndian, l.LearningRate)
	}
}

// next layer length
func (c *Chain) writeBiasInfo(file *os.File) {
	for _, l := range *c.Layers {
		for _, n := range *l.Bias {
			binary.Write(file, binary.BigEndian, n)
		}
	}
}

func (c *Chain) writeWeightInfo(file *os.File) {
	for _, l := range *c.Layers {
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
