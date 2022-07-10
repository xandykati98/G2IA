import { NeuralNetwork, scaleBetween } from './nn_organized';
// 5000 train, 2500 test
console.log('Importando dados de: mnist.train.json')
import train_set from './mnist/train.json';
console.log('Importação finalizada')

console.log('Importando dados de: mnist.test.json')
import test_set from './mnist/test.json';
console.log('Importação finalizada')
const train = train_set as { data: number[]; label: string }[];
const test = test_set as { data: number[]; label: string }[];

const rede = new NeuralNetwork()

rede.pushLayer({
    is_input: true,
    neurons_number: 13
})

rede.pushLayer({
    neurons_number: 26,
})
rede.pushLayer({
    neurons_number: 26,
})

rede.pushLayer({
    is_output: true,
    neurons_number: 10,
})

rede.createWeights()
const label_to_output = (label: string) => {
    const output = new Array(10).fill(0)
    output[parseInt(label)] = 1
    return output
}
const normalized_train_set = train.map(item => ({ 
    inputs: item.data
    .map(num => num/255)
    .filter((_, i) => i % 2 === 0)
    .filter((_, i) => i % 2 !== 0)
    .filter((_, i) => i % 2 === 0)
    .filter((_, i) => i % 2 !== 0), 
    desired_outputs: label_to_output(item.label) 
}))
console.log(normalized_train_set[0].inputs.length)
rede.train({
    taxa_aprendizado: 0.01,
    epochs: 1000,
    iteracoes: 10,
    debug: true,
    training_set: normalized_train_set
})
/**
let v_i = 0;

let acertos = 0
while (v_i < test.length) {
    const test_item = test[v_i]
    const test_item_inputs = test_item.data.map(num => scaleBetween(num, 0, 1, 0, 255))
    const test_item_outputs = rede.guess(test_item_inputs)
    const test_item_label = test_item.label
    const test_item_label_output = label_to_output(test_item_label)

    if (test_item_label_output.toString() === test_item_outputs.toString()) {
        acertos++
    }
    console.log(`Acertos: ${acertos} de ${test.length}`)
    v_i++
}
console.log(`Acertos: ${acertos} de ${test.length}`) */