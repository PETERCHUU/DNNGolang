from pezzza

# Alternating least squares

one of the rainforest model

## als use for ai self train and make it can be train and use in real time

https://www.youtube.com/watch?v=EvV5Qtp_fYg

## each layer have three step

> Layer N

### normal forwarding step

each neuron output float point for next layer

### Evaluation

using each float point use fitness function output score

### selection

shorting the score, use first 30% of score and directly output float point to next layer

### mutation

other 70& randomly use different function to change the neuron,

1. Nothing
2. NewConnection
3. New neuron add in middle
4. Change Weight and Bias

> Layer N+1

# network work flow

1. just making one layer for input , and one neuron for output layer,don't connect them
2. change weight on neuron
3. model find bad score, add connection to output layer
4. model find bad score, add more neuron in middle
5. add connection between new neuron and other neuron
6. loop step 2 to 5 until _confidence_ >= your expected
