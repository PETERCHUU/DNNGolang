package FileReader

import (
	"encoding/binary"
	"io"
	"os"
)

type MnstSample struct {
	Label [10]float64  // nn output
	Image [784]float64 // nn input
}

func InitSample(imageFilePath, LabelFilePath string) []MnstSample {
	// 打开文件
	imageFile, err := os.Open(imageFilePath)
	labelFile, err := os.Open(LabelFilePath)
	if err != nil {
		panic(err)
	}
	defer imageFile.Close()
	var b int32
	var length int32
	for i := 0; i < 4; i++ {
		err := binary.Read(imageFile, binary.BigEndian, &b)
		if err != nil {
			panic(err)
		}
		if i == 1 {
			length = b
		}
		//fmt.Printf("Byte value: %d\n", b)

	}
	for i := 0; i < 2; i++ {

		err := binary.Read(labelFile, binary.BigEndian, &b)
		if err != nil {
			panic(err)
		}
		//fmt.Printf("Byte value: %d\n", b)

	}
	var ImageBuffer = make([]uint8, 784)
	var LabelBuffer byte
	var Sample = make([]MnstSample, length)
	i := 0

	for {

		_, err := imageFile.Read(ImageBuffer)
		if err != nil && err != io.EOF {
			panic(err)
		}
		err = binary.Read(labelFile, binary.BigEndian, &LabelBuffer)
		if err != nil && err != io.EOF {
			panic(err)
		}
		label := [10]float64{}
		switch LabelBuffer % 10 {
		case 0:
			label[0] = 1
		case 1:
			label[1] = 1
		case 2:
			label[2] = 1
		case 3:
			label[3] = 1
		case 4:
			label[4] = 1
		case 5:
			label[5] = 1
		case 6:
			label[6] = 1
		case 7:
			label[7] = 1
		case 8:
			label[8] = 1
		case 9:
			label[9] = 1
		}

		Sample[i].Label = label

		for j := 0; j < 784; j++ {
			Sample[i].Image[j] = float64(ImageBuffer[j])
		}

		if err == io.EOF || i >= int(length)-1 {
			break
		}
		i++
	}
	return Sample
}
