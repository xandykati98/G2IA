import { benchmark } from "./benchmark"
import { NeuralNetwork, sigmoidRandom, TrainConfig } from "./nn_organized"
// import data
import all_data from './data2.json'
import { exit } from "process"

let rede = new NeuralNetwork()

// create dataset
let data = all_data.map(d => {
    const { label, id, dormitórios, suítes, vagas, caracteristicas, fotos_qtd, destaque, preço_venda, preço_locação } = d
    return {
        inputs: [label, dormitórios, suítes, vagas, caracteristicas, fotos_qtd, destaque, preço_venda, preço_locação],
        desired_outputs: [label]
    }
})



rede.pushLayer({
    is_input: true,
    neurons_number: data[0]['inputs'].length,
})

rede.pushLayer({
    neurons_number:9,
    inline_bias: true,
    activation_function: 'relu'
})
rede.pushLayer({
    neurons_number:9,
    inline_bias: true,
    activation_function: 'relu'
})
rede.pushLayer({
    neurons_number:9,
    inline_bias: true,
    activation_function: 'relu'
})

rede.pushLayer({
    is_output: true,
    neurons_number: 1,
    activation_function: 'linear'
})

rede.initAllWeights()

// split data 80% train, 20% test
const [train_data, test_data] = data.reduce(([train, test], d) => {
    if (Math.random() < 0.8) {
        // @ts-ignore
        train.push(d)
    } else {
        // @ts-ignore
        test.push(d)
    }
    return [train, test]
} , [[], []])


const train_config:TrainConfig = {
    epochs: 60,
    iteracoes: 10000,
    taxa_aprendizado: 0.01,
    training_set: train_data,
    momentum: 0.9,
}

benchmark({
    v_set: test_data, 
    train_config: train_config, 
    get_prediction: output => output > 0.5 ? 1 : 0, 
    rede: rede, 
    runs:1,
    bechnmark_name: "regression",
})
