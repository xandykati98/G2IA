# G2IA
 
## Introduction
First we need to define our NeuralNetwork structure. 
```ts
// Create a new neural network
const net = new NeuralNetwork()

// Append layers to the network
net.pushLayer({
    // Since this is the first layer, it is the input layer
    is_input: true, 
    neurons_number: 2,
})

// Append a hidden layer
net.pushLayer({
    neurons_number: 3,
})

// Append a output layer
net.pushLayer({
    is_output: true,
    neurons_number: 1,
})
```
Before we can train our network we need to instance the layers' weights.
```ts
// This will initialize the weights of the network with random values.
net.initAllWeights()

// or, we could initialize the weights on a specific layer
net.initWeights(layer, layer_index)
```
## Training
```ts
const train_set:any[] = [
    { inputs: [0, 0], desired_outputs: [0] },
    { inputs: [0, 1], desired_outputs: [1] },
    { inputs: [1, 0], desired_outputs: [1] },
    { inputs: [1, 1], desired_outputs: [0] },
]

const train_config = {
    epochs: 90,
    momentum: 0.01,
    iteracoes: 10000, // iterations per epoch
    taxa_aprendizado: 0.01, // learning rate
    training_set: train_set
}

const train_result = net.train(train_config)

const { 
    epochs,
    mean_error,
    std_error,
    min_error,
    max_error,
    error_diff,
    last_error,
    taxa_aprendizado,
    iteracoes_por_epoch
} = train_result;
```

## Testing & Prediction
```ts
const test_set:any[] = [
    { inputs: [0, 0], desired_outputs: [0] },
    { inputs: [0, 1], desired_outputs: [1] },
    { inputs: [1, 0], desired_outputs: [1] },
    { inputs: [1, 1], desired_outputs: [0] },
]

for (const { inputs, desired_outputs } of test_set) {
    const output = net.guess(test_set)
    
    console.log(`${desired_outputs} --- ${output}`)
}

```