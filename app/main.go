package main

func main() {

	i := 0

	defer func() {
		if r := recover(); r != nil {
			println("recover")
			i--
			println(i)
		}
	}()
	for i < 10 {
		testPanic(i)
		i++
	}

	println(i)
}
func testPanic(i int) {
	if i > 4 {
		panic("panic")
	}
	println(i)
}
