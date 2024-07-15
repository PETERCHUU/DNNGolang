[![Go Reference](https://pkg.go.dev/badge/github.com/PETERCHUU/DNNGolang.svg)](https://pkg.go.dev/github.com/PETERCHUU/DNNGolang)

This is a flux like DNN module library that build by using standard library only

it very easy to use

```go
package main


import (
	"github.com/PETERCHUU/DNNGolang"
	"github.com/PETERCHUU/DNNGolang/Mnist"
	"github.com/PETERCHUU/DNNGolang/Mnist/FileReader"
)



func main(){
    //making a 700 floating point input , 1 layer of hidden with 50 plating point, and final 10 point of output
    module:= DNNGolang.NewNetwork().FCLayer(700, 50, function.Sigmoid, learningRate).
		FCLayer(50, 10, function.Softmax, learningRate)

    // get a 700 length 2D array for input use
    array:=make([]float64,700)
    for _,l:= range array{
        
    }
    module.predict()
}

```
