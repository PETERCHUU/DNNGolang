run:
	go run $(file)

testfunction:
	go test -v -run $(function) $(file)

test:
	go test -v 

testfile:
	go test -v $(file)

build:
	go build -o $(file) $(file)